"""The engine API: the only door between the client (``astro_gpu/client``) and the engine.

Every function takes the project definition (``Project2``) first, and a workflow by its integer id (a ``Workflow``
object also works wherever the function only reads: that is how what-ifs are asked). Edits change the definition in
memory; ``save(p)`` writes it (the client autosaves, debounced). Nothing here imports Qt.

Threading. Nothing slow runs on import: torch, the stages and the renderers load on the first call that
needs them. The calls fall into four groups:

* memory only (the GUI thread may call them): the editing functions (``add_workflow``, ``set_option``, ``set_param``,
  ``set_workflows``, ``make_variant``...), ``workflows``, ``methods``, ``sliders_of``, ``running``, ``cancel``;
* the stage graph and the keys: ``chain``, ``deps``, ``blocks``, ``stage_key``, ``sharing``, ``states``, ``plan_*``,
  ``cost``, ``estimate``. The first call imports the stages (seconds); the parts of the keys that read the disk (the
  frame files of the sessions, the reference image) come from an ``InputScan`` cached per project, so only the first
  call after ``open_project`` (or after ``refresh(p, scan=True)``) scans. Call ``warm(p)`` from the worker thread
  when a project opens; the GUI thread then finds everything cached. ``states`` reads manifests (small files);
* worker thread only (they scan, load arrays or compute): ``warm``, ``scan_folder``, ``add_frames``, ``health_checks``,
  ``frame_table``, ``selection_preview``, ``frame_thumb``, ``describe``, ``stage_outputs``, ``stage_array``,
  ``cache_usage``, ``clear_unused``, ``machine``, ``list_projects``, ``import_v1`` (preview and import), and the
  diagnostics of the basic stages: ``calibration_frames``, ``calibrated_light``, ``registration_table``,
  ``stack_diagnostics``, ``color_fit``;
* the render worker: ``render_layer``, ``post_masks``, ``export_image``, ``export_workflow``.

A run is a detached process (``start_run``); the client follows ``read_events`` and calls ``refresh(p)`` when a stage
ends so that cached "done" answers are read again. ``states(p)`` records the done keys in ``state.json`` (the "previous
result" of a stage that later goes Outdated); ``states(p, update=False)`` writes nothing: use it on a project that must
stay read-only.

For tests and tools, ``use_rules(p, rules)`` replaces the stage graph (toy stages); the spec ``'module:function'``
also reaches the detached worker through ``Plan.rules``.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import re
import shutil
import tempfile
import threading
import weakref
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .core import devices
from .core import estimate as est
from .core import project2 as p2
from .core import workflow as wfl
from .core.project import Target, type_warnings
from .core.project2 import KINDS, Project2, Session, Workflow
from .core.skyobjects import CatalogueUnavailable, SkyObject  # noqa: F401 - re-exported: sky_objects raises it
from .core.store import InputScan, KeyRules, Keys, Store, StoreProject

if TYPE_CHECKING:
    import numpy as np

    from .core.import_v1 import ImportReport
    from .core.runner2 import Plan, RunInfo
    from .render.layers import LayerImage

TEMPLATES = wfl.TEMPLATES
OptionRefused = wfl.OptionRefused
LegacyProject = p2.LegacyProject          # open_project raises it on a v1 folder
Estimate = est.Estimate

# reasons of the stage states (English; the client translates them)
REASON_NEVER = "never run"
REASON_SETTINGS = "its settings changed since this result"
REASON_UPSTREAM = "an upstream result changed since this result"
REASON_ALGORITHM = "the algorithm was updated since this result"
REASON_UPSTREAM_FAILED = "upstream failed"
REASON_INTERRUPTED = "interrupted"
STATES = ("done", "pending", "stale", "running", "failed", "interrupted", "skipped")


class NotDone(FileNotFoundError):
    """A result that is asked for is not computed (yet): the message names the stage."""


class RunInProgress(RuntimeError):
    """The project has a live run: what was asked would pull the files away from it."""


# ---------------------------------------------------------------------------------------------- return types

@dataclass
class ProjectSummary:
    """A Home card. ``state``: empty (no workflow), never_run, has_results (cheap scan) or, with ``deep``,
    up_to_date, outdated, running."""

    name: str
    root: str
    description: str = ""
    target: str = ""
    nights: int = 0
    workflows: int = 0
    integration_h: float | None = None
    last: str = ""                       # ISO time of the last change (definition, state or events)
    state: str = "empty"
    done: int = 0                        # workflows with results
    outdated: int | None = None          # deep only
    running: bool = False
    thumb: str = ""                      # cache/thumb.jpg when there is one
    error: str = ""                      # the definition cannot be read


@dataclass
class ScanResult:
    """What a folder holds: the frames of each kind (the folder itself holds the lights) and the calibration folders
    found next to it."""

    folder: str
    name: str = ""
    target: str = ""
    folders: dict[str, list[str]] = field(default_factory=dict)         # kind -> folders
    frames: dict[str, list[Any]] = field(default_factory=dict)          # kind -> [io.frames.FrameInfo]
    errors: list[str] = field(default_factory=list)                     # unreadable files


@dataclass
class GroupingCriteria:
    """What splits the frames into sessions: the night (noon to noon), the exposure, the ISO or gain."""

    night: bool = True
    exposure: bool = True
    iso: bool = True


@dataclass
class SessionGroup:
    name: str
    night: str
    exposure: float | None
    iso: float | None
    paths: list[str]
    start: str = ""
    end: str = ""
    filter: str = ""                  # the filter the FITS headers name (a monochrome camera's): never stacked with another

    @property
    def count(self) -> int:
        return len(self.paths)


@dataclass
class Check:
    """A health check: ``level`` ok | warn | info; ``key`` a code; ``text`` English; ``args`` the numbers in it."""

    level: str
    key: str
    text: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameRow:
    index: int
    path: str
    name: str
    session: int | None
    time: str | None
    fwhm: float
    transparency: float
    rms: float
    weight: float
    status: str                          # '' kept, else the reason ("low transparency", "excluded by hand"...)
    excluded: bool                       # excluded by hand (the project's list)


@dataclass
class FrameTable:
    rows: list[FrameRow]
    kept: int
    min_transparency: float
    max_fwhm_ratio: float
    source: str                          # "stack" (what the stack used) or "preview" (these thresholds)

    @property
    def total(self) -> int:
        return len(self.rows)


@dataclass
class MethodInfo:
    """A way to compute a block. ``new``: not in the engine (listed, disabled)."""

    key: str
    name: str
    short: str
    runs_on: str                         # gpu | cpu | both
    stage: str
    new: bool = False
    what: str = ""


@dataclass
class BlockRef:
    block: str
    stages: tuple[str, ...]
    key: tuple[str, ...]


@dataclass
class StageState:
    """``state`` is one of STATES. ``previous_key``: the result an Outdated stage had, still in the store."""

    state: str
    reason: str = ""
    key: str = ""
    previous_key: str | None = None


@dataclass
class Cost:
    seconds: float
    measured: bool                       # every time comes from this machine's history (else ≈)
    disk_gb: float
    jobs: int
    keys: list[str] = field(default_factory=list)
    skipped: dict[int, str] = field(default_factory=dict)


@dataclass
class Gpu:
    device: str                          # "cuda:0", "xpu:0", "mps"
    name: str
    memory_gb: float
    beta: bool = False                   # an Intel, AMD or Apple card: never tested here, the card test guards it


@dataclass
class MachineInfo:
    gpus: list[Gpu] = field(default_factory=list)
    cores: int = 0
    memory_gb: float = 0.0

    @property
    def cuda(self) -> bool:
        return bool(self.gpus)


@dataclass
class CacheItem:
    """Something in the store or the scratch folder that no workflow needs. ``kind``: result, tmp, trash, scratch.
    ``resumable``: a ``.tmp`` folder of the key a workflow has now (its checkpoints let the next run resume)."""

    kind: str
    stage: str
    key: str
    path: str
    bytes: int
    resumable: bool = False


@dataclass
class CacheUsage:
    used_bytes: int
    unused: list[CacheItem] = field(default_factory=list)

    @property
    def unused_bytes(self) -> int:
        return sum(i.bytes for i in self.unused)


@dataclass
class StageOutputs:
    """A done result: its folder, previews {name: jpg path}, and the manifest's metrics, parameters, origins and
    reasons as they are (the reasons are the engine's templates; ``manifest`` is the whole file)."""

    stage: str
    key: str
    which: str                           # current | previous
    dir: str
    previews: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    origin: dict[str, str] = field(default_factory=dict)
    auto_reasons: dict[str, Any] = field(default_factory=dict)
    duration: float | None = None
    finished: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParamInfo:
    """A setting of a stage for the panel. ``value`` is the effective one; ``origin`` default | auto | user;
    ``suggested`` the value without the user's (``suggestion_known`` False: the stage cannot compute it yet, the
    upstream results are not done). ``reason`` / ``reason_args``: the engine's ``Auto.template`` and ``Auto.args``,
    unformatted (``reason_en`` is ``Auto.reason``). ``drift``: the done result used another value
    (``result_value``); ``locked``: the workflow's options or the project's exclusions set it. ``advice``: the value
    the engine advises without applying it (the stack's "method": "drizzle" or "classic"; None when it gives none or
    cannot compute it yet), its reason in ``advice_reason`` / ``advice_args`` (template and arguments, as for
    ``reason``) and ``advice_reason_en``."""

    name: str
    label: str
    help: str
    tip: str
    type: str
    default: Any
    value: Any
    suggested: Any
    suggestion_known: bool
    origin: str
    reason: str = ""
    reason_args: Any = None
    reason_en: str = ""
    level: str = "basic"
    unit: str = ""
    ge: float | None = None
    le: float | None = None
    gt: float | None = None
    lt: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    locked: bool = False
    drift: bool = False
    result_value: Any = None
    advice: Any = None
    advice_reason: str = ""
    advice_args: Any = None
    advice_reason_en: str = ""


@dataclass
class SliderInfo:
    key: str
    name: str
    group: str
    ends: tuple[str, str]
    available: bool
    reason: str = ""


# the diagnostics of the basic stages (calibration_frames, registration_table, stack_diagnostics, color_fit). Each
# result lists in ``missing`` what a result of an older stage version did not keep, as "<what>: missing: recompute
# <stage> to get it" (the field is then None / empty).

@dataclass
class CalFrame:
    """A frame of a session as the calibration sees it. ``kind``: darks, flats, bias, flat_darks or lights.
    ``stats``: raw ADU, black level not subtracted (``core/diagnostics.raw_stats``): median, planes (the level of each
    Bayer position), noise, hot_pixels, rings (radial profile, centre to corners), centre, edge, gradient (centre -
    edge: "growing central light" when positive), saturated_pct, max, and exposure, iso, temperature (°C, None when the
    file does not say), time; flats also level_pct (% of full scale above ``zero``, their offset), vignetting (each
    ring relative to the centre) and corner_pct; lights the calibrated frame's median, noise, planes and noise_planes
    (the stage's frame_stats, in the cache's ADU). ``flag`` ok | warn | bad, ``reason`` in plain words (warn: an
    outlier against the other frames of its kind, a badly exposed flat; bad: not a frame of its kind at all, a dark
    with stars or far above the camera's black level; nothing is excluded), ``checks`` the same as
    [{"key", "text", numbers}]. ``calibrated``: the calibrated cache file of a light, ``index`` its frame there
    (``calibrated_light`` reads it); '' when the cache is gone."""

    kind: str
    path: str
    name: str
    stats: dict[str, Any] = field(default_factory=dict)
    flag: str = "ok"
    reason: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    calibrated: str = ""
    index: int | None = None


@dataclass
class CalibrationReport:
    """``masters`` {kind: .npy path} of the calibration result (dark, flat, bias, flat_dark, bad_pixels: the ones it
    has), made from the frames of all the sessions; ``frames`` the session's calibration frames, then its lights;
    ``notes`` plain-word remarks; ``stage_dir`` the result's folder ('' when none is done)."""

    session: int
    masters: dict[str, str] = field(default_factory=dict)
    frames: list[CalFrame] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    stage_dir: str = ""


@dataclass
class RegRow:
    """A light frame as registration aligned it (``core/diagnostics.frame_geometry`` for dx, dy, rotation, scale; NaN
    when not aligned). ``nmatch``: stars matched to the reference, ``nstars``: stars detected, ``fwhm`` px,
    ``roundness`` minor / major axis of the stars (1 = round; None for a result that did not keep it). ``used``: the
    frame was aligned (the stack may still leave it out: ``stack_diagnostics``); ``reason`` when not."""

    name: str
    time: str | None
    dx: float
    dy: float
    rotation: float
    scale: float
    nmatch: int
    nstars: int
    fwhm: float
    roundness: float | None
    used: bool
    index: int = 0
    path: str = ""
    rms: float = float("nan")            # residual of the alignment (px)
    transparency: float = float("nan")   # star flux relative to the reference frame
    reference: bool = False
    reason: str = ""
    # register's Moffat fit on the green samples (FWHM px, roundness minor / major, beta); None for a registration
    # result made before it
    fwhm_moffat: float | None = None
    roundness_moffat: float | None = None
    beta_moffat: float | None = None


class RegTable(list):
    """``registration_table``'s answer: a list of RegRow, with ``missing`` and ``notes``."""

    def __init__(self, rows: Iterable[RegRow] = (), missing: Iterable[str] = (), notes: Iterable[str] = ()):
        super().__init__(rows)
        self.missing = list(missing)
        self.notes = list(notes)


@dataclass
class StackDiag:
    """What the stack did. ``rejection_low`` / ``rejection_high``: (h, w) counts of the samples the second pass
    rejected below / above the prediction, on the stack grid reduced ``rejection_factor`` times on each axis (None:
    missing). ``weights``: (name, weight, norm_offset, norm_scale) of every frame, the normalisation being value ->
    (value - offset) x scale per channel R, G, B (offset: the frame's sky gradient at the field centre, ADU; scale:
    1 / transparency); None for a frame left out or a result that did not keep it. ``frames``: the full record of
    each used frame (index, path, weight, transparency, offset, scale, gradient: the 6 coefficients per channel of
    the quadratic sky difference, sky, half A | B, rejected_low, rejected_high: on the stack, rejected_outside: the
    rejected samples that land outside it). ``noise`` {R, G, B}: pixel noise of the result (ADU of the stack), from
    its two half-stacks. ``used``: the frames stacked; ``excluded`` {name: reason}. ``rejected_pct``: the stage's
    metric (since version 4 of the stack, the share of the samples landing on the stack that were rejected; before, it
    counted the samples landing outside it too); ``rejected_on_grid_pct``: the share of the samples landing on the
    stack that were rejected."""

    rejection_low: Any = None
    rejection_high: Any = None
    rejection_factor: int = 1
    weights: list[tuple[str, float, tuple[float, ...] | None, tuple[float, ...] | None]] = field(default_factory=list)
    noise: dict[str, float] = field(default_factory=dict)
    used: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    frames: list[dict[str, Any]] = field(default_factory=list)
    rejected_pct: float | None = None
    rejected_on_grid_pct: float | None = None
    grid: tuple[int, int] | None = None              # (width, height) of the stack
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # ---- the details of the stacking (filled for every stack; what a switch adds is there only when it was on)
    #: the switches the stack was made with ({} for a stack made before the stack kept them)
    switches: dict[str, str] = field(default_factory=dict)
    #: {"method": "own" (or as recorded), "rejected_pct", "untested_pct" (Hubble: fewer than 3 samples around),
    #: "reference_pct": Siril's mark [0.1, 0.5], "prediction" (Hubble: samples per point and median / minmed per colour),
    #: "classic" and "reason_template" (the Classic method: the rejection its rule chose, {"template", "args"})}
    rejection: dict[str, Any] = field(default_factory=dict)
    #: (h, w) counts of the samples left untested (Hubble), on the rejection maps' grid; None otherwise
    rejection_untested: Any = None
    #: the coverage per colour: "maps" (3, h, w) the drizzle weight map reduced ``factor`` times (block means),
    #: "variation" {R, G, B} its std / median (DrizzlePac's statistic, under 20 % is even), and when kept
    #: "estimate" (core/coverage's fast estimate at the scale and drop used) and "rule" (the coverage rule's scale,
    #: drop and method, each {value, template, args, reason})
    coverage: dict[str, Any] = field(default_factory=dict)
    #: per frame name, its weight in each colour (R, G, B)
    weights_rgb: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    #: per excluded frame name, why: [{"code", "template", "args"}] (texts.format_reason translates them)
    exclusions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: the frame selection and weights (core/frame_weights): the filters' medians, sigmas and limits, the costs, the
    #: guarded weights, the automatic reasons; None for a stack made before the stack kept them
    selection: dict[str, Any] | None = None
    #: the local normalisation recorded by stacks of an earlier engine version (mode, reference frames, stars,
    #: diagnostic, reason); None otherwise
    normalisation: dict[str, Any] | None = None
    #: the colour alignment applied: "mode" ("night": one correction measured on the mean, or as recorded)
    #: and its summary
    channel_alignment: dict[str, Any] | None = None


@dataclass
class StarFit:
    """A catalogue star of the colour calibration. ``x``, ``y``: stack pixels. ``measured``: its colour in the camera,
    (R, G, B) fractions of its flux (None: not measured). ``catalogue``: the CIE xy chromaticity of its Gaia XP
    spectrum, which the fit aims at (``bp_rp``: its Gaia BP - RP). ``predicted``: the xy the fitted matrix gives
    from the measured colour; ``residual`` their distance (None: not used). ``used``: in the matrix fit; ``reason``
    why not, in plain words. ``camera_predicted``: the (R, G, B) fractions the fitted sensitivity curves predict
    from the spectrum; ``outlier``: the channels whose curve fit left the star out (e.g. "B")."""

    x: float
    y: float
    measured: tuple[float, float, float] | None
    catalogue: tuple[float, float]
    predicted: tuple[float, float] | None
    residual: float | None
    used: bool
    reason: str = ""
    source: int = 0
    g: float = float("nan")
    bp_rp: float | None = None
    flux: tuple[float, float, float] | None = None
    camera_predicted: tuple[float, float, float] | None = None
    outlier: str = ""


@dataclass
class ColorFit:
    """The colour calibration's fit: ``stars``, ``matrix`` (3, 3) camera -> linear sRGB as applied (M.npy), ``stats``
    (stars, used, rejected {reason: count}, chromaticity_residual and residual_p90 of the used stars, response_source,
    fit_residual, fit_agreement_pct, matrix_xyz: the fitted camera -> XYZ matrix up to its scale, aperture,
    white_temp, computed: "stage" or "again" (measured again on the stack for an older result))."""

    stars: list[StarFit] = field(default_factory=list)
    matrix: Any = None
    stats: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------- per-project cache

class _NullJournal:
    """Where a resolve writes its events: nowhere."""

    def emit(self, *a: Any, **k: Any) -> None:
        pass


class _Ctx:
    """What the API keeps for one project object: the cached disk parts of the keys, the Keys of the current
    definition, the store's folders and the automatic values already computed."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.scan = InputScan()
        self.rules_obj: KeyRules | None = None
        self.spec: str | None = None
        self.fp: str | None = None
        self.keys: Keys | None = None
        self.store: Store | None = None
        self.store_root: str | None = None
        self.autos: dict[tuple, dict] = {}
        self.stretch: dict[tuple, tuple[float, float, float]] = {}
        self.warm_entries: set[tuple] = set()

    def rules(self) -> KeyRules:
        if self.rules_obj is None:
            from .core.store import key_rules
            self.rules_obj = key_rules()
        return self.rules_obj


_CTXS: dict[int, _Ctx] = {}
_CTX_LOCK = threading.Lock()


def _ctx(p: Project2) -> _Ctx:
    k = id(p)
    c = _CTXS.get(k)
    if c is None:
        with _CTX_LOCK:
            c = _CTXS.get(k)
            if c is None:
                c = _CTXS[k] = _Ctx()
                weakref.finalize(p, _CTXS.pop, k, None)
    return c


def _keys(p: Project2) -> Keys:
    """The Keys of the project's current definition: a snapshot (so a worker thread never sees a half-made edit), made
    again when the definition changes. Its ``project`` is the snapshot."""
    c = _ctx(p)
    d = p.to_dict()
    seen = {k: v for k, v in d.items() if k not in ("current_block", "show_all")}   # view state keys nothing
    fp = hashlib.sha1(json.dumps([str(p.root), seen], sort_keys=True, default=str).encode()).hexdigest()
    with c.lock:
        if c.keys is None or c.fp != fp:
            c.keys = Keys(Project2.from_dict(d, p.root), c.rules(), c.scan)
            c.fp = fp
        return c.keys


def _store(p: Project2) -> Store | None:
    if p.root is None:
        return None
    c = _ctx(p)
    root = os.path.abspath(p.root)
    with c.lock:
        if c.store is None or c.store_root != root:
            c.store, c.store_root = Store(root), root
        return c.store


def _r2() -> Any:
    from .core import runner2
    return runner2


def _load_rules(rules: KeyRules | str | None) -> KeyRules | None:
    """A KeyRules, or the one a spec ``'module:function'`` builds."""
    if isinstance(rules, str):
        mod, _, fn = rules.partition(":")
        return getattr(importlib.import_module(mod), fn)()
    return rules


def use_rules(p: Project2, rules: KeyRules | str | None, spec: str | None = None) -> None:
    """Replace the stage graph of this project (tests and tools): ``rules`` is a KeyRules, or a ``'module:function'``
    spec building one. ``spec`` names the same rules for the detached worker (``Plan.rules``)."""
    c = _ctx(p)
    with c.lock:
        if isinstance(rules, str):
            c.rules_obj, c.spec = _load_rules(rules), rules
        else:
            c.rules_obj, c.spec = rules, spec
        c.fp = None
        c.autos.clear()


def refresh(p: Project2, scan: bool = False) -> None:
    """Forget what was cached about the disk: the 'usable' answers and the automatic values (after a run or a stage
    end). ``scan=True`` also forgets the frame files and file ids (the raws changed on disk)."""
    c = _ctx(p)
    with c.lock:
        if c.store is not None:
            c.store.forget()
        c.autos.clear()
        if scan:
            c.scan.forget()
            c.warm_entries.clear()
        c.fp = None


def warm(p: Project2, progress: Callable[[float, str], None] | None = None) -> None:
    """Fill every disk cache the keys, states and plans use (worker thread): the stage graph, each session's frame
    files, the reference image, the keys of every workflow, the sensor size the estimates need and the 'usable'
    answers of the done inventory and calibrate results. Afterwards those calls read manifests only."""
    say = progress or (lambda f, m: None)
    say(0.0, "loading the stage graph")
    keys = _keys(p)
    store = _store(p)
    c = _ctx(p)
    ws = keys.project.workflows
    for i, w in enumerate(ws):
        say(0.1 + 0.8 * i / max(len(ws), 1), f"keys of {w.name}")
        for s in keys.chain(w):
            k = keys.key(w, s)
            if store is not None and s in ("inventory", "calibrate") and store.done(s, k):
                store.usable_as_input(s, k)
        est.facts(keys, w, store)                       # reads one raw header (the sensor size) for the estimates
    with c.lock:
        c.warm_entries = {_entries(keys.project, s) for s in keys.project.sessions}
    say(1.0, "ready")


def _entries(project: Project2, session: Session) -> tuple:
    return tuple(tuple(project.abspath(f) for f in getattr(session, k)) for k in KINDS)


def is_warm(p: Project2) -> bool:
    """Whether the frame files of every session are cached (``warm`` ran, and the sessions did not change since)."""
    c = _ctx(p)
    return all(_entries(p, s) in c.warm_entries for s in p.sessions) if p.root is not None else True


# ---------------------------------------------------------------------------------------------- projects

def list_projects(folder: str | Path, deep: bool = False) -> list[ProjectSummary]:
    """The projects of a folder, most recent first. Skips ``_*`` folders and folders without
    ``project.astroproj``; a definition that cannot be read gives a summary with ``error``. A cheap scan: the state is
    read from state.json and the store (never the raws); ``deep=True`` computes the true states (worker thread)."""
    ws = Path(folder)
    out: list[ProjectSummary] = []
    if not ws.is_dir():
        return out
    for d in sorted(ws.iterdir()):
        if d.is_dir() and not d.name.startswith("_") and (d / p2.FILE).is_file():
            out.append(_summary(d, deep))
    out.sort(key=lambda s: s.last, reverse=True)
    return out


def project_summary(root: str | Path, deep: bool = False) -> ProjectSummary:
    """The summary of one project folder (as ``list_projects`` gives each)."""
    return _summary(Path(root), deep)


def _mtime(*paths: Path) -> float:
    best = 0.0
    for f in paths:
        try:
            best = max(best, f.stat().st_mtime)
        except OSError:
            pass
    return best


def _lock_alive(root: Path) -> bool:
    try:
        holder = json.loads((root / "logs" / "run.lock").read_text(encoding="utf-8"))
        import psutil
        return psutil.pid_exists(int(holder["pid"]))
    except Exception:  # noqa: BLE001 - no lock, an unreadable one, no psutil: no live run
        return False


def _summary(d: Path, deep: bool) -> ProjectSummary:
    last = datetime.fromtimestamp(_mtime(d / p2.FILE, d / p2.STATE_FILE, d / "logs" / "events.jsonl"))
    s = ProjectSummary(name=d.name, root=str(d), last=last.isoformat(timespec="seconds"))
    try:
        p = p2.load(d)
    except Exception as e:  # noqa: BLE001 - a broken definition still gets its card
        s.error = f"{type(e).__name__}: {e}"
        return s
    s.name, s.description, s.target = p.name, p.description, p.target.name
    s.nights, s.workflows = sum(1 for x in p.sessions if x.lights), len(p.workflows)
    s.running = _lock_alive(d)
    thumb = d / "cache" / "thumb.jpg"
    s.thumb = str(thumb) if thumb.is_file() else ""
    store = Store(d)
    last_keys = p2.read_state(d)["last_keys"]

    def done(stage: str, key: Any) -> bool:
        try:
            return store.done(stage, key)
        except ValueError:                       # not a key: state.json was edited by hand
            return False
    for w in p.workflows:
        if any(done(st, k) for st, k in (last_keys.get(str(w.id)) or {}).items()):
            s.done += 1
    for lk in last_keys.values():
        k = lk.get("inventory")
        m = (store.manifest("inventory", k).get("metrics") or {}) if k and done("inventory", k) else {}
        if m.get("n_lights") and m.get("exposure_s"):
            s.integration_h = round(m["n_lights"] * m["exposure_s"] / 3600, 1)
            break
    s.state = "empty" if not p.workflows else ("has_results" if s.done else "never_run")
    if deep and p.workflows:
        try:
            st = states(p, update=False)
        except Exception as e:  # noqa: BLE001
            s.error = f"{type(e).__name__}: {e}"
            return s
        bad = {wid for (wid, _), x in st.items() if x.state != "done"}
        s.outdated = len(bad)
        s.state = "running" if s.running else ("never_run" if not s.done else ("outdated" if bad else "up_to_date"))
    return s


def create_project(root: str | Path, name: str, target: Target | dict | str | None = None) -> Project2:
    """A new, empty v2 project folder (no sessions, no workflows). ``target``: a Target, its fields, or its name."""
    if isinstance(target, str):
        target = Target(name=target)
    elif isinstance(target, dict):
        target = Target(**target)
    return p2.create(root, name, target)


def open_project(root: str | Path) -> Project2:
    """Open a v2 project (nothing is written). A v1 folder raises ``LegacyProject``: import it (``import_v1``)."""
    return p2.load(root)


def save(p: Project2) -> Path:
    """Write the definition atomically; returns the file."""
    return p2.save(p)


def save_as(p: Project2, root: str | Path) -> Project2:
    """Copy the definition to a new folder (definition only: no results) and return that project; ``p`` stays where
    it was. Relative paths become absolute, so that the copy still finds the frames and the reference image."""
    root = Path(root)
    if (root / p2.FILE).exists():
        raise FileExistsError(f"a project already exists in {root}")
    q = Project2.from_dict(p.to_dict(), None)
    for s in q.sessions:
        for kind in KINDS:
            setattr(s, kind, [p.abspath(f) for f in getattr(s, kind)])
    if q.reference_image:
        q.reference_image = p.reference_path()
    if q.scratch_dir:
        q.scratch_dir = p.abspath(q.scratch_dir)
    q.created = datetime.now().isoformat(timespec="seconds")
    p2.save(q, root)
    return q


def import_v1(old_root: str | Path, new_root: str | Path, path_map: dict[str, Any] | None = None,
              dry_run: bool = False, *, rules: KeyRules | str | None = None,
              progress: Callable[[str], None] | None = None) -> ImportReport:
    """Import a v1 project and its results into a new v2 folder (copy only; nothing is ever written into the old
    project). ``path_map`` says where the raw frames are now ({old folder or file: new place, or a list of places}).
    ``dry_run=True`` is the preview: the same checks, nothing written, the report says per workflow and stage what is
    adopted (with its key) or recomputed (why), the bytes to copy and the free space (``report.enough_space``).
    The real import raises ``NotEnoughSpace`` before it copies anything when the disk is too small, and
    ``WriteRefused`` if anything tried to write into the old project. ``progress(text)`` gets short messages.
    ``rules``: toy stages for tests (a KeyRules, or the spec ``'module:function'``). Worker thread: it reads every
    frame's header and copies tens of GB. Open the result with ``open_project(new_root)``."""
    from .core import import_v1 as iv
    return iv.import_v1(old_root, new_root, path_map, dry_run, rules=_load_rules(rules), progress=progress)


# ---------------------------------------------------------------------------------------------- frames and sessions

def scan_folder(folder: str | Path, progress: Callable[[float, str], None] | None = None) -> ScanResult:
    """Read a folder of light frames and the calibration folders found next to it (darks, flats, bias, dark flats,
    by their usual names): every frame with its header. Slow on a big card: worker thread. ``progress(fraction,
    text)`` may raise to stop. What it finds is a suggestion: ``scan_files`` reads what the user chooses instead."""
    from .core.newproject import guess_calibration
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    return scan_files({"lights": [str(folder)], **guess_calibration(str(folder))}, progress)


def scan_files(chosen: dict[str, list[str | Path]], progress: Callable[[float, str], None] | None = None) -> ScanResult:
    """Read the frames the user chose, kind by kind (``KINDS``: lights, darks, flats, bias, flat_darks), each a list of
    folders (every frame in them) or of files: every frame with its header. The files that make up a whole folder are
    kept as that folder (the project lists folders, as an import of folders makes). Worker thread; ``progress`` as
    ``scan_folder``."""
    from concurrent.futures import ThreadPoolExecutor

    from .core.newproject import guess_names
    from .core.project import list_frames
    from .io.frames import read_info
    say = progress or (lambda f, m: None)
    entries = {k: [str(x) for x in (chosen.get(k) or [])] for k in KINDS}
    first_light = next((Path(x) for x in entries["lights"]), None)
    where = first_light if first_light is None or first_light.is_dir() else first_light.parent
    names = guess_names(str(where)) if where is not None else {"name": "", "target": ""}
    res = ScanResult(str(where or ""), names["name"], names["target"])
    res.folders = {k: _whole_folders(v) for k, v in entries.items()}
    files = {k: list_frames(fs) for k, fs in res.folders.items()}
    total, done = max(sum(len(v) for v in files.values()), 1), 0
    with ThreadPoolExecutor(8) as ex:
        for kind, fl in files.items():
            res.frames[kind] = []
            if not fl:
                continue
            first = None
            for f in fl:                                  # the first readable frame is decoded; the others by header
                try:
                    first = read_info(f)
                    break
                except Exception as e:  # noqa: BLE001
                    res.errors.append(f"{f}: {e}")
            if first is None:
                continue

            def read(f: Path, like=first) -> Any:
                try:
                    return read_info(f, like=like)
                except Exception as e:  # noqa: BLE001 - one bad file does not stop the scan
                    return f"{f}: {e}"
            for r in ex.map(read, fl):
                (res.frames[kind].append(r) if not isinstance(r, str) else res.errors.append(r))
                done += 1
                say(done / total, f"reading headers {done}/{total}")
    return res


def _whole_folders(entries: list[str]) -> list[str]:
    """Files that make up every frame of their folder become that folder; the others stay files; folders stay."""
    from .core.project import list_frames
    out: list[str] = []
    by_dir: dict[str, list[str]] = {}
    for e in entries:
        p = Path(e)
        if p.is_file():
            by_dir.setdefault(str(p.parent), []).append(str(p))
        else:
            _put(out, e)
    for d, fl in by_dir.items():
        every = {os.path.normcase(str(x)) for x in list_frames([d])}
        if every and every == {os.path.normcase(x) for x in fl}:
            _put(out, d)
        else:
            for x in sorted(fl):
                _put(out, x)
    return out


def _when(info: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(info.time).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def group_sessions(infos: Sequence[Any], criteria: GroupingCriteria | None = None) -> list[SessionGroup]:
    """Split frames (``FrameInfo``) into sessions: one per night (noon to noon), then by exposure and ISO / gain as
    the criteria say. Frames of different filters (a monochrome camera's FITS: Ha, OIII...) are never one session,
    whatever the criteria: a session stacks one filter. Groups come in time order."""
    c = criteria or GroupingCriteria()
    groups: dict[tuple, list[Any]] = {}
    for i in infos:
        t = _when(i)
        night = (t - timedelta(hours=12)).date().isoformat() if t else "unknown"
        key = (night if c.night else "", i.exposure if c.exposure else None, i.iso if c.iso else None,
               getattr(i, "filter", "") or "")
        groups.setdefault(key, []).append(i)
    out: list[SessionGroup] = []
    order = sorted(groups.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1],
                                                   -1 if kv[0][2] is None else kv[0][2], kv[0][3]))
    for (night, exp, iso, flt), fl in order:
        times = sorted(t.isoformat(timespec="seconds") for t in map(_when, fl) if t)
        name = night or "all frames"
        more = sum(1 for k in groups if k[0] == night and k[3] == flt) > 1
        if more and c.exposure and exp is not None:
            name += f" · {exp:g} s"
        if more and c.iso and iso is not None:
            name += f" · ISO {iso:g}"
        if flt:                                       # the filter in the name: two filters of a night, two names
            name += f" · {flt}"
        out.append(SessionGroup(name, night, exp, iso, sorted(i.path for i in fl),
                                times[0] if times else "", times[-1] if times else "", flt))
    return out


def unique_session_name(p: Project2, name: str) -> str:
    """``name``, or ``name (2)``, ``name (3)``... when a session of the project has it (two sessions never share a
    name)."""
    taken = {s.name for s in p.sessions}
    out, k = name, 2
    while out in taken:
        out, k = f"{name} ({k})", k + 1
    return out


def add_session(p: Project2, lights: Iterable[str | Path] = (), darks: Iterable[str | Path] = (),
                flats: Iterable[str | Path] = (), bias: Iterable[str | Path] = (),
                flat_darks: Iterable[str | Path] = (), name: str = "") -> Session:
    """A new session from folders or files (no header is read); its name is made unique."""
    sid = max((s.id for s in p.sessions), default=0) + 1
    s = Session(sid, unique_session_name(p, name or f"Session {sid}"), lights=[str(x) for x in lights],
                darks=[str(x) for x in darks], flats=[str(x) for x in flats], bias=[str(x) for x in bias],
                flat_darks=[str(x) for x in flat_darks])
    p.sessions.append(s)
    p.validate()
    return s


def _same(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def _put(lst: list[str], item: str) -> None:
    if not any(_same(item, x) for x in lst):
        lst.append(item)


def add_frames(p: Project2, scan: ScanResult, session: int | None = None) -> list[Session]:
    """Add a scanned folder to the project. Without ``session``, one session per night is made (noon to noon); the
    calibration folders go to the session nearest in time. With a session id, everything goes into it. Returns the
    sessions made or changed."""
    lights = scan.frames.get("lights") or []
    if session is not None:
        s = p.session(session)
        for kind in KINDS:
            for f in scan.folders.get(kind, []):
                if scan.frames.get(kind):
                    _put(getattr(s, kind), f)
        p.validate()
        return [s]
    groups = group_sessions(lights, GroupingCriteria(night=True, exposure=False, iso=False)) if lights else []
    if not groups:
        groups = [SessionGroup(scan.name or Path(scan.folder).name, "", None, None, [])]
    made: list[Session] = []
    whole = len(groups) == 1
    for g in groups:
        night = g.night if g.night and g.night != "unknown" else ""
        s = add_session(p, name=(night + (f" · {g.filter}" if g.filter else "")) if night else g.name)
        s.lights = list(scan.folders["lights"]) if whole else list(g.paths)
        s.filter = g.filter                       # read from the FITS headers (a monochrome camera's filter)
        made.append(s)
    mids = [datetime.fromisoformat(g.start) if g.start else None for g in groups]
    for kind in KINDS[1:]:
        for f in scan.folders.get(kind, []):
            fl = [i for i in scan.frames.get(kind, []) if _same(str(Path(i.path).parent), f) or _same(str(i.path), f)]
            if not fl:
                continue
            ts = sorted(t for t in map(_when, fl) if t)
            flt = next((i.filter for i in fl if getattr(i, "filter", "")), "")
            near = [i for i, g in enumerate(groups) if g.filter == flt] if flt else []
            near = near or list(range(len(made)))     # flats of a filter go to a session of that filter
            if whole or not ts or not any(mids[i] for i in near):
                target = made[near[0]]
            else:
                mid = ts[len(ts) // 2]
                target = made[min((i for i in near if mids[i]), key=lambda i: abs(mids[i] - mid))]
            night = groups[made.index(target)].night
            for i, s in enumerate(made):              # darks and bias without a filter: every filter of that night
                if s is target or (not flt and groups[i].filter and groups[i].night == night):
                    _put(getattr(s, kind), f)
    p.validate()
    return made


def remove_session(p: Project2, session: int) -> Session:
    """Remove a session and the frames of it excluded by hand; returns it."""
    s = p.session(session)
    p.sessions.remove(s)
    p.excluded = [pair for pair in p.excluded if pair[0] != session]
    p.validate()
    return s


def set_excluded(p: Project2, pairs: Iterable[tuple[int, str]], excluded: bool = True) -> None:
    """Exclude (or keep again) light frames by hand: ``(session id, file name)`` pairs. Only Stacking and the stages
    after it change (the stack's ``exclude`` setting)."""
    pairs = [(int(sid), str(name)) for sid, name in pairs]
    for sid, _ in pairs:
        p.session(sid)
    cur = list(p.excluded)
    if excluded:
        cur += [t for t in dict.fromkeys(pairs) if t not in cur]
    else:
        cur = [t for t in cur if t not in set(pairs)]
    p.excluded = cur
    p.validate()


def health_checks(p: Project2) -> list[Check]:
    """What to look at before running: no lights, no bias, flats of several sessions merged into one master, the star
    size per session (once Registration is done) and the disk the calibrated cache needs. Worker thread (it reads
    the frame lists and one header)."""
    out: list[Check] = []
    inp = p.inputs()
    n = {k: len(getattr(inp, k)) for k in KINDS}
    if not p.sessions or not n["lights"]:
        return [Check("warn", "no_lights", "No light frames yet")]
    from .io.frames import colour_session
    keys = _keys(p)
    for s in keys.project.sessions:                  # issue #58: a file added by mistake, said before the run
        fr = keys.scan.session_frames(keys.project, s)
        for w in type_warnings({k: [str(f) for f, _size in fr[k]] for k in KINDS}):
            out.append(Check("warn", "file_types", type_warning_english(w, s.name), {**w, "session": s.name}))
        note = colour_session([str(f) for f, _size in fr["lights"]])
        if note:                                     # issue #57: the limit of such frames, said plainly
            out.append(Check("info", "colour_frames", f"{s.name}: {note['type']} frames already in colour: no Bayer "
                             "drizzle; dark and flat correction are approximate when they are not linear data",
                             {**note, "session": s.name}))
    for w in p.workflows:                            # two filters in one stack: a mistake, said before the run
        flts = sorted({s.filter for s in p.sessions_of(w) if s.lights} - {""}) if w.source is None else []
        if len(flts) > 1:
            out.append(Check("warn", "filters_mixed", f"{w.name}: frames of {', '.join(flts)} would be stacked "
                             "together; give each filter its own workflow", {"workflow": w.name, "filters": flts}))
    if not n["bias"]:
        out.append(Check("info", "no_bias", "No bias frames"))
    with_flats = [s for s in p.sessions if s.flats]
    if len(with_flats) > 1:
        out.append(Check("warn", "flats_merged", f"The flats of {len(with_flats)} sessions are merged into one master "
                         "flat", {"n": len(with_flats), "sessions": [s.name for s in with_flats]}))
    try:
        for name, v in _fwhm_per_session(p).items():
            out.append(Check("info", "fwhm", f"Star size in {name}: {v:.1f} px", {"session": name, "fwhm": v}))
    except Exception:  # noqa: BLE001 - a hint: not measured yet
        pass
    out.append(_disk_check(p))
    if not any(c.level == "warn" for c in out):
        out.insert(0, Check("ok", "ok", "Nothing to worry about"))
    return out


def type_warning_english(w: dict, session: str) -> str:
    """A file-type warning (core.project.type_warnings) in English."""
    if w["key"] == "unread":
        return (f"{session}: {w['name']} is not a frame the engine reads here (camera raw or FITS; a JPEG, PNG or TIFF "
                "only with lights of its kind)" if w["n"] == 1 else
                f"{session}: {w['n']} {w['type']} files are not frames the engine reads here (camera raw or FITS; "
                "a JPEG, PNG or TIFF only with lights of its kind)")
    if w["key"] == "mixed":
        return f"{session}: {w['n']} {w['type']} file(s) among {w['of']} {w['main']} {w['kind']}: added by mistake?"
    if w["key"] == "same_files":
        return f"{session}: {w['n']} file(s) of the {w['kind']} are also in the {w['other']}"
    if w["key"] == "wrong_kind":
        return f"{session}: the {w['kind']} are {w['type']}, the lights {w['main']}: they cannot calibrate them"
    return f"{session}: the {w['kind']} are {w['type']}, the lights {w['main']}"


def _disk_check(p: Project2) -> Check:
    keys = _keys(p)
    lights = [f for s in keys.project.sessions for f, _ in keys.scan.session_frames(keys.project, s)["lights"]]
    scratch = str(keys.project.scratch)
    if not lights:
        return Check("info", "disk", "No light frames to size the cache on")
    mpx = est._sensor_mpx(str(lights[0]))
    need = len(lights) * (mpx or 0) * 1e6 * 2 / 1e9
    probe = Path(scratch)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free / 1e9
    args = {"need_gb": round(need, 1), "free_gb": round(free)}
    text = f"Calibrated cache: about {need:.1f} GB on the scratch disk ({free:.0f} GB free)."
    if need > free * 0.9:
        return Check("warn", "disk", text + " Not enough space!", args)
    return Check("ok", "disk", text, args)


def _fwhm_per_session(p: Project2) -> dict[str, float]:
    import numpy as np
    for w in p.workflows:
        if _dir(p, w.id, "register") is not None:
            t = frame_table(p, w.id)
            by: dict[int, list[float]] = {}
            for r in t.rows:
                if r.session is not None and np.isfinite(r.fwhm):
                    by.setdefault(r.session, []).append(r.fwhm)
            return {p.session(sid).name: float(np.median(v)) for sid, v in by.items()}
    return {}


# ---------------------------------------------------------------------------------------------- frame table, thumbs

def _light_sessions(p: Project2) -> dict[str, int]:
    """{normalised path: session id} of the light frames (the first session that lists a file)."""
    keys = _keys(p)
    out: dict[str, int] = {}
    for s in keys.project.sessions:
        for f, _ in keys.scan.session_frames(keys.project, s)["lights"]:
            out.setdefault(os.path.normcase(os.path.normpath(str(f))), s.id)
    return out


def _dir(p: Project2, wf: int | Workflow, stage: str, which: str = "current") -> Path | None:
    """The folder of a done result of the workflow's stage (``which`` current, or previous: the result an Outdated
    stage had), else None. Issue #43: ``which`` may name a session ("current@2": that session's instance of the
    stage, for its Session and Calibration blocks); a stage the chain does not have is read from the stage that stands
    for it (the sessions' join for "inventory" and "calibrate", the image for "stack")."""
    store = _store(p)
    if store is None:
        return None
    keys = _keys(p)
    w = keys.project.workflow(wf)
    which, _at, sid = which.partition("@")
    chain = keys.chain(w)
    if sid and f"{stage}@{sid}" in chain:
        stage = f"{stage}@{sid}"
    elif stage not in chain:
        stage = keys.rules.chains.resolve(chain, "", stage) or stage
    if stage not in chain:
        return None
    k = keys.key(w, stage)
    if which == "previous":
        k = (p2.read_state(p.root)["last_keys"].get(str(w.id)) or {}).get(stage)
        if not k or k == keys.key(w, stage):
            return None
    return store.dir(stage, k) if store.done(stage, k) else None


def _need(p: Project2, wf: int | Workflow, stage: str, which: str = "current") -> Path:
    d = _dir(p, wf, stage, which)
    if d is None:
        raise NotDone(f"the {stage} result of workflow {_wid(wf)} is not computed ({which})")
    return d


def _wid(wf: int | Workflow) -> int:
    return wf.id if isinstance(wf, Workflow) else wf


def _table(p: Project2, wf: int | Workflow, min_t: float | None, max_f: float | None, source: str) -> FrameTable:
    import numpy as np
    inv = json.loads((_need(p, wf, "inventory") / "frames.json").read_text(encoding="utf-8"))["lights"]
    reg = np.load(_need(p, wf, "register") / "reg.npz")
    n = len(reg["fwhm"])
    stack = _dir(p, wf, "stack") if source == "stack" else None
    source = "stack" if stack is not None else "preview"
    sess = _light_sessions(p)
    ex = set(p.excluded)
    names = [Path(f["path"]).name for f in inv[:n]]
    sid = [sess.get(os.path.normcase(os.path.normpath(f["path"]))) for f in inv[:n]]
    by_hand = np.array([(s, nm) in ex for s, nm in zip(sid, names)], bool)
    if stack is not None:
        sf = json.loads((stack / "frames.json").read_text(encoding="utf-8"))
        weight = np.array([float(f["weight"]) for f in sf])
        reasons = [str(f.get("excluded") or "") for f in sf]
        min_t, max_f = _stack_thresholds(p, wf)
    else:
        from .stages.stack import StackParams, frame_selection
        stats = np.load(_need(p, wf, "calibrate") / "frame_stats.npy")
        infos = {i.name: i for i in describe(p, wf, "stack")}
        if min_t is None or max_f is None:
            d_min, d_max = float(infos["min_transparency"].value), float(infos["max_fwhm_ratio"].value)
            min_t, max_f = d_min if min_t is None else min_t, d_max if max_f is None else max_f
        # the stack's own selection: the thresholds act only with the "thresholds" rule; by default the frames
        # under the minimum weight are left out (stages/stack.frame_selection)
        sp = StackParams(min_transparency=min_t, max_fwhm_ratio=max_f, selection=infos["selection"].value,
                         selection_transparency=infos["selection_transparency"].value)
        sel = frame_selection(sp, reg, stats, by_hand)
        weight = sel.weight
        reasons = [str(r) for r in sel.reasons]
    rows = [FrameRow(i, inv[i]["path"], names[i], sid[i], inv[i].get("time"), float(reg["fwhm"][i]),
                     float(reg["transp"][i]), float(reg["rms"][i]), float(weight[i]), reasons[i], bool(by_hand[i]))
            for i in range(n)]
    return FrameTable(rows, sum(1 for r in rows if not r.status), float(min_t), float(max_f), source)


def _stack_thresholds(p: Project2, wf: int | Workflow) -> tuple[float, float]:
    """The stack's selection thresholds as it has them now (user, automatic or default)."""
    infos = {i.name: i for i in describe(p, wf, "stack")}
    return float(infos["min_transparency"].value), float(infos["max_fwhm_ratio"].value)


def frame_table(p: Project2, wf: int | Workflow) -> FrameTable:
    """One row per light frame, with its star size, transparency, registration residual, weight and the reason it is
    left out (what the stack used when Stacking is done, else the selection the current thresholds give). Needs
    Registration done (``NotDone`` otherwise)."""
    return _table(p, wf, None, None, "stack")


def selection_preview(p: Project2, wf: int | Workflow, min_transparency: float,
                      max_fwhm_ratio: float) -> FrameTable:
    """The frame table the given thresholds would give (Selection's sliders), with the frames excluded by hand; the
    thresholds are measured on all the frames, as the stack does."""
    return _table(p, wf, float(min_transparency), float(max_fwhm_ratio), "preview")


def _mtf(m: float, x: Any) -> Any:
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def _superpixel(mosaic: Any, cfa: str, black: float) -> Any:
    """The mosaic as (h/2, w/2, 3): each 2x2 cell is one pixel (the green cells averaged; X-Trans: core/cfa's), black
    level removed."""
    import numpy as np

    from .core.cfa import superpixel
    a = np.asarray(mosaic, np.float32) - np.float32(black)
    if cfa == "RGB":                                 # a frame already in colour, its three planes (issue #57)
        return superpixel(a, cfa)
    H, W = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    if not cfa:
        g = a[:H, :W].reshape(H // 2, 2, W // 2, 2).mean((1, 3))
        return np.repeat(g[:, :, None], 3, axis=2)
    return superpixel(a, cfa)


def frame_thumb(p: Project2, path: str | Path, size: int = 256) -> Any:
    """A thumbnail of a raw frame: (h, w, 3) uint8, its long side ``size``. One stretch per session (measured on its
    middle frame), so that blinking the frames of a session shows real differences. Worker thread."""
    import cv2
    import numpy as np

    from .io.frames import read_cfa, read_colour, read_info

    def mosaic(f, cfa):                              # a frame already in colour: its planes, for a colour thumbnail
        return read_colour(f) if cfa == "RGB" else read_cfa(f)

    path = Path(path)
    info = read_info(path)
    img = _superpixel(mosaic(path, info.cfa), info.cfa, info.black)
    sid = _light_sessions(p).get(os.path.normcase(os.path.normpath(str(path))))
    st = None
    if sid is not None:
        keys = _keys(p)
        s = keys.project.session(sid)
        ck = (sid, _entries(keys.project, s))
        c = _ctx(p)
        st = c.stretch.get(ck)
        if st is None:
            files = [f for f, _ in keys.scan.session_frames(keys.project, s)["lights"]]
            ref = files[len(files) // 2]
            ri = read_info(ref)
            st = c.stretch[ck] = _stretch_of(_superpixel(mosaic(ref, ri.cfa), ri.cfa, ri.black))
    if st is None:
        st = _stretch_of(img)
    bp, wp, mb = st
    x = np.clip((img - bp) / (wp - bp), 0, 1)
    y = np.clip(_mtf(mb, x), 0, 1)
    f = size / max(y.shape[:2])
    if f < 1:
        y = cv2.resize(y, (max(1, round(y.shape[1] * f)), max(1, round(y.shape[0] * f))),
                       interpolation=cv2.INTER_AREA)
    return (y * 255 + 0.5).astype(np.uint8)


def _stretch_of(img: Any) -> tuple[float, float, float]:
    """(black point, white point, midtone balance) putting the sky of a superpixel image at 0.25."""
    import numpy as np
    y = img.mean(2)[::2, ::2]
    m = float(np.median(y))
    sig = 1.4826 * float(np.median(np.abs(y - m)))
    bp = m - 2.8 * sig
    wp = float(np.percentile(y, 99.9))
    if wp <= bp + 1e-6:
        wp = bp + 1.0
    mn = float(np.clip((m - bp) / (wp - bp), 1e-4, 0.99))
    return bp, wp, float(_mtf(0.25, mn))


# ---------------------------------------------------------------------------------------------- workflows

def _ids(p: Project2, wf_ids: int | Iterable[int]) -> list[int]:
    ids = [wf_ids] if isinstance(wf_ids, int) else list(wf_ids)
    for i in ids:
        p.workflow(i)
    return list(dict.fromkeys(ids))


def workflows(p: Project2) -> list[Workflow]:
    """The workflows (copies: changing them changes nothing; ``set_workflows`` takes such a list back)."""
    return copy.deepcopy(p.workflows)


def set_current(p: Project2, wf: int) -> None:
    p.workflow(wf)
    p.current = wf


def _options_of(options: dict[str, Any] | str | None) -> dict[str, Any] | None:
    if isinstance(options, str):
        if options not in wfl.TEMPLATES:
            raise ValueError(f"unknown workflow template {options!r} (known: {list(wfl.TEMPLATES)})")
        return dict(wfl.TEMPLATES[options])
    return dict(options) if options is not None else None


def add_workflow(p: Project2, name: str, options: dict[str, Any] | str | None = None,
                 params: dict[str, dict[str, Any]] | None = None, copy_of: int | None = None) -> Workflow:
    """A new workflow: from a template (its name, or the options), or a copy of another (``copy_of``: its options,
    stored settings and exit; ``options`` and ``params`` then change the copy). Returns a copy of it."""
    opts = _options_of(options)
    end = None
    merged: dict[str, dict[str, Any]] = {}
    if copy_of is not None:
        src = p.workflow(copy_of)
        opts = {**src.options, **(opts or {})}
        merged = copy.deepcopy(src.params)
        end = src.end
    if opts is None:
        opts = dict(wfl.DEFAULT_OPTIONS)
    for st, kv in (params or {}).items():
        merged.setdefault(st, {}).update(copy.deepcopy(kv))
    w = p.add_workflow(name, opts, merged, end=end)
    if copy_of is not None:                            # the same frames: its Calibration placed in parallel too
        w.calibration = copy.deepcopy(p.workflow(copy_of).calibration)
    return copy.deepcopy(w)


def rename_workflow(p: Project2, wf: int, name: str) -> None:
    if not str(name).strip():
        raise ValueError("a workflow needs a name")
    p.workflow(wf).name = str(name).strip()


def duplicate_workflow(p: Project2, wf: int, name: str | None = None) -> Workflow:
    """A copy of a workflow (options and stored settings), named "<name> copy" unless ``name`` is given."""
    src = p.workflow(wf)
    return add_workflow(p, name or f"{src.name} copy", None, None, copy_of=wf)


def delete_workflow(p: Project2, wf: int) -> Workflow:
    """Remove a workflow (its results stay in the store until the cache is cleared); the workflows that were variants
    of it become plain workflows. Returns it."""
    w = p.remove_workflow(wf)
    for o in p.workflows:
        if o.variant_of == wf:
            o.variant_of = None
    return copy.deepcopy(w)


def set_workflows(p: Project2, workflows: list[Workflow], current: int | None = None) -> None:
    """Put the workflows back as a list taken earlier (Undo / Redo). Ids are never reused: a workflow deleted since
    comes back with its own id."""
    old = (p.workflows, p.next_workflow_id, p.current)
    try:
        p.workflows = copy.deepcopy(list(workflows))
        p.next_workflow_id = max([p.next_workflow_id] + [w.id + 1 for w in p.workflows])
        ids = [w.id for w in p.workflows]
        p.current = current if current in ids else (p.current if p.current in ids else (ids[0] if ids else None))
        p.validate()
    except Exception:
        p.workflows, p.next_workflow_id, p.current = old
        raise


def set_option(p: Project2, wf_ids: int | Iterable[int], key: str, value: Any) -> list[str]:
    """Add or remove an optional block, or change a method, in these workflows (the rules of ``workflow.set_option``).
    Returns the notes on what came or went with it (toasts). Raises ``OptionRefused`` and changes nothing when a
    workflow's rules forbid it."""
    new: dict[int, dict[str, Any]] = {}
    notes: list[str] = []
    for wid in _ids(p, wf_ids):
        w = p.workflow(wid)
        new[wid], n = wfl.set_option(w.options, key, value, w.end)
        notes += [x for x in n if x not in notes]
    for wid, o in new.items():
        w = p.workflow(wid)
        w.options = o
        if w.blocks is not None:                     # a graph (issue #47) follows
            p.set_blocks(w, wfl.sync_blocks(w.blocks, wfl.source_of(w), o))
    return notes


def _check_param(keys: Keys, stage: str, name: str, value: Any) -> Any:
    """The value as the stage's Params read it (JSON form); ValueError naming the setting if it is not valid."""
    stages = keys.rules.chains.stages
    if wfl.base(stage) not in stages:
        raise ValueError(f"unknown stage {stage!r}")
    model = keys.rules.params(wfl.base(stage))      # a session's own settings: "calibrate@2" (issue #43)
    if name not in model.model_fields:
        raise ValueError(f"{stage} has no setting {name!r}")
    if (model.model_fields[name].json_schema_extra or {}).get("level") == "internal":
        raise ValueError(f"{stage}.{name} is set by the project, not by the user")
    try:
        return model(**{name: value}).model_dump(mode="json")[name]
    except ValueError as e:
        from pydantic import ValidationError
        msg = "; ".join(x["msg"] for x in e.errors()) if isinstance(e, ValidationError) else str(e)
        raise ValueError(f"{stage}.{name}: {msg}") from None


def _suggestion(p: Project2, w: Workflow, stage: str, name: str) -> tuple[bool, Any]:
    """(known, value): what the setting would be without the user's value (see ``describe``)."""
    keys = _keys(p)
    if name not in keys.rules.auto_fields(stage):
        return True, keys.rules.defaults(stage).get(name)
    r = _resolve(p, w, stage)
    return r.known[name], r.suggested[name]


def _params_with(p: Project2, w: Workflow, stage: str, name: str, value: Any) -> dict[str, dict[str, Any]]:
    """The stored settings of ``w`` once ``stage.name`` is set to ``value``. None goes back to the suggestion; a value
    equal to the suggestion is dropped only when the suggestion is known (the upstream results are done, or the
    setting has no automatic rule): otherwise it stays stored, as an override."""
    keys = _keys(p)
    v = None if value is None else _check_param(keys, stage, name, value)
    params = copy.deepcopy(w.params)
    d = params.setdefault(stage, {})
    drop = v is None
    if not drop:
        known, sug = _suggestion(p, w, stage, name)
        drop = known and sug == v
    if drop:
        d.pop(name, None)
    else:
        d[name] = v
    if not d:
        params.pop(stage, None)
    return params


def set_param(p: Project2, wf_ids: int | Iterable[int], stage: str, name: str, value: Any = None) -> None:
    """Store a setting in these workflows (``sharing()`` gives the ones that share a block). ``value`` None goes
    back to the suggestion. A value equal to the suggestion is dropped only when the suggestion is known; while the
    upstream results are not done it stays stored (typing the default is how a user overrides an automatic value).
    Raises ValueError naming the setting when the value is not valid."""
    keys = _keys(p)
    new = {wid: _params_with(p, keys.project.workflow(wid), stage, name, value) for wid in _ids(p, wf_ids)}
    for wid, params in new.items():
        p.workflow(wid).params = params


# a change is {"kind": "opt", "key", "value"} or {"kind": "param", "stage", "name", "value"}
Change = dict[str, Any]


def _changed(p: Project2, w: Workflow, change: Change) -> tuple[Workflow, list[str]]:
    """A copy of ``w`` with the change made, and the notes."""
    c = copy.deepcopy(w)
    notes: list[str] = []
    if change.get("kind") == "opt":
        c.options, notes = wfl.set_option(c.options, change["key"], change["value"], c.end)
    elif change.get("kind") == "param":
        c.params = _params_with(p, c, change["stage"], change["name"], change.get("value"))
    else:
        raise ValueError(f"unknown change {change!r}: kind is 'opt' or 'param'")
    return c, notes


def with_change(p: Project2, wf: int | Workflow, change: Change) -> Workflow:
    """What-if: a copy of the workflow with the change made (it is not in the project; ``cost`` and ``stage_key`` take
    it)."""
    return _changed(p, _keys(p).project.workflow(wf), change)[0]


def apply_change(p: Project2, wf_ids: int | Iterable[int], change: Change) -> list[str]:
    """Make the change in these workflows (Replace); returns the notes."""
    keys = _keys(p)
    done = {wid: _changed(p, keys.project.workflow(wid), change) for wid in _ids(p, wf_ids)}
    notes: list[str] = []
    for wid, (c, n) in done.items():
        w = p.workflow(wid)
        w.options, w.params = c.options, c.params
        notes += [x for x in n if x not in notes]
    return notes


def throws_away(p: Project2, wf_ids: int | Iterable[int], change: Change) -> bool:
    """Whether the change would make a done result of these workflows Outdated (then Replace or Try as variant)."""
    keys, store = _keys(p), _store(p)
    if store is None:
        return False
    for wid in _ids(p, wf_ids):
        w = keys.project.workflow(wid)
        c, _ = _changed(p, w, change)
        now = set(keys.chain(c))
        for s in keys.chain(w):
            if store.done(s, keys.key(w, s)) and (s not in now or keys.key(c, s) != keys.key(w, s)):
                return True
    return False


def make_variant(p: Project2, wf: int, change: Change, name: str) -> Workflow:
    """Try as variant: a copy of the workflow with the change, linked to it (``variant_of``); the original stays as it
    is. Returns the variant."""
    src = p.workflow(wf)
    c, _ = _changed(p, _keys(p).project.workflow(wf), change)
    w = p.add_workflow(name, c.options, c.params, end=c.end, variant_of=src.id)
    return copy.deepcopy(w)


def keep_variant(p: Project2, variant: int) -> Workflow:
    """Keep: the variant replaces its original and takes its name (the original's id is retired). Returns it."""
    v = p.workflow(variant)
    if v.variant_of is None:
        raise ValueError(f"workflow {variant} is not a variant")
    orig = p.workflow(v.variant_of)
    v.name = orig.name
    v.variant_of = orig.variant_of
    for o in p.workflows:
        if o.variant_of == orig.id and o.id != v.id:
            o.variant_of = v.id
    p.workflows.remove(orig)
    p.current = v.id
    p.validate()
    return copy.deepcopy(v)


def discard_variant(p: Project2, variant: int) -> None:
    """Discard: remove the variant; its original becomes the current workflow."""
    v = p.workflow(variant)
    if v.variant_of is None:
        raise ValueError(f"workflow {variant} is not a variant")
    parent = v.variant_of
    p.remove_workflow(variant)
    for o in p.workflows:
        if o.variant_of == variant:
            o.variant_of = parent
    p.current = parent
    p.validate()


def add_exit(p: Project2, wf: int, block: str, name: str) -> Workflow:
    """A workflow that copies another up to ``block`` (Stacking), then goes to its own simple output. Only the
    settings of the stages it runs are kept."""
    src = p.workflow(wf)
    if block not in wfl.EXITS:
        raise ValueError(f"no early output at {block!r} (allowed: {list(wfl.EXITS)})")
    chain = wfl.chain(src.options, block, reference=bool(p.reference_image))
    params = {s: copy.deepcopy(kv) for s, kv in src.params.items() if s in chain}
    return copy.deepcopy(p.add_workflow(name, dict(src.options), params, end=block))


_METHODS: dict[str, list[MethodInfo]] = {
    "denoise": [
        MethodInfo("pretrained", "Pretrained network", "pretrained", "both", "denoise", False,
                   "A network trained on many pairs of half-stacks (Noise2Noise), applied as it is. Nothing is "
                   "invented: it learned to predict one half-stack from the other. A monochrome stack of fewer than "
                   "30 frames gets a network made for monochrome noise. It slightly brightens faint stars (+8 to +11 % "
                   "on the bench): it is for pictures, not for photometry."),
        MethodInfo("adapted", "Pretrained + adapted to the image", "adapted", "both", "denoise", False,
                   "The same network first learns this image's own noise on its half-stacks: longer, fitted to this "
                   "image (a session without dithering most of all)."),
    ],
    "gradient": [
        MethodInfo("polynomial", "Polynomial", "polynomial", "cpu", "gradient", False,
                   "A smooth surface fitted on the sky between the objects, removed right after stacking."),
        MethodInfo("rbf", "RBF", "RBF", "cpu", "gradient", False,
                   "A surface through the samples of sky (thin-plate splines): follows a gradient of any shape; best "
                   "with samples placed by hand."),
        MethodInfo("automatic", "Automatic", "automatic", "cpu", "gradient", False,
                   "A flexible surface fitted on every pixel that does not look like an object: follows complex "
                   "gradients, may take part of the faintest wide structures."),
    ],
    "stars": [
        MethodInfo("neural", "Neural star removal", "neural", "both", "star_removal", False,
                   "Takes the stars off the image (a network trained on many stacks): the starless image and the stars "
                   "alone, combined by a Recombine block or each used as a layer of Post. Seconds on a "
                   "GPU, a few minutes on a processor."),
    ],
    "deconvolution": [
        MethodInfo("rl", "Measured PSF (Richardson-Lucy)", "Richardson-Lucy", "both", "deconv", True,
                   "Classic deconvolution with the PSF measured across the field."),
        MethodInfo("neural", "Neural network", "neural", "both", "deconv", True,
                   "A pretrained network that corrects the PSF locally, stars included."),
    ],
}


def methods(block: str) -> list[MethodInfo]:
    """The methods of a block (denoise, gradient, stars, deconvolution). ``new`` marks the ones that are not in the
    engine yet; ``key`` is the workflow option value that selects it; ``stage`` the stage that implements it."""
    if block not in _METHODS:
        raise ValueError(f"block {block!r} has no methods (known: {list(_METHODS)})")
    return copy.deepcopy(_METHODS[block])


def chain(p: Project2, wf: int | Workflow) -> list[str]:
    """The stages of a workflow, in order."""
    return list(_keys(p).chain(wf))


def deps(p: Project2, wf: int | Workflow, stage: str) -> tuple[str, ...]:
    """What a stage reads in this workflow (optional and extra dependencies only when present)."""
    return _keys(p).deps(wf, stage)


def blocks(p: Project2, wf: int | Workflow) -> list[BlockRef]:
    """The blocks of a workflow in graph order: ``stages`` in its chain, ``key`` the tuple of their keys (Astrometry
    leaves out the survey; Output adds the workflow id; Selection has no stage and so an empty key)."""
    keys = _keys(p)
    return [BlockRef(b, tuple(ss), keys.block_key(wf, b))
            for b, ss in keys.rules.chains.blocks_in(keys.chain(wf))]


# ---------------------------------------------------------------------------------------------- keys, states, cache

def stage_key(p: Project2, wf: int | Workflow, stage: str) -> str:
    return _keys(p).key(wf, stage)


def sharing(p: Project2, wf: int | Workflow, stage: str) -> list[int]:
    """The workflows that compute this stage exactly as ``wf`` does (``wf`` included)."""
    keys = _keys(p)
    w = keys.project.workflow(wf)
    k = keys.key(w, stage)
    out = [o.id for o in keys.project.workflows if o.id != w.id and stage in keys.chain(o) and keys.key(o, stage) == k]
    return sorted(out + [w.id])


def _why_stale(keys: Keys, store: Store, w: Workflow, stage: str, previous: str) -> str:
    old = store.manifest(stage, previous).get("key_parts")
    try:
        cur = json.loads(json.dumps(keys.key_parts(w, stage)))
    except ValueError:
        return REASON_SETTINGS
    if not isinstance(old, dict):
        return REASON_UPSTREAM
    if old.get("v") != cur.get("v"):
        return REASON_ALGORITHM
    if old.get("p") != cur.get("p") or old.get("x") != cur.get("x"):
        return REASON_SETTINGS
    return REASON_UPSTREAM


def states(p: Project2, update: bool = True) -> dict[tuple[int, str], StageState]:
    """The state of every stage of every workflow: done, pending (never run), stale (Outdated: it has a previous
    result in the store), running, failed, interrupted, skipped (an upstream stage failed, or the workflow is blocked).
    Reads manifests, not the raws. ``update=True`` writes ``state.json``: the key of every done stage, the "previous
    result" of a stage that later goes Outdated."""
    keys, store, r2 = _keys(p), _store(p), _r2()
    proj = keys.project
    root = p.root
    last = p2.read_state(root) if root is not None else {"last_keys": {}}
    last_keys: dict[str, dict[str, str]] = last["last_keys"]
    new_last = {w: dict(kv) for w, kv in last_keys.items()}
    run = r2.running(root) if root is not None else None
    live: set[str] = set()
    if run is not None:
        try:
            live = {j["key"] for j in json.loads(Path(run.plan_file).read_text(encoding="utf-8")).get("jobs") or []}
        except (OSError, ValueError):
            pass
    out: dict[tuple[int, str], StageState] = {}
    for w in proj.workflows:
        chain_ = keys.chain(w)
        bad: set[str] = set()
        prev_keys = last_keys.get(str(w.id)) or {}
        for s in chain_:
            try:
                k = keys.key(w, s)
            except ValueError as e:                    # a stored setting is not valid
                out[(w.id, s)] = StageState("failed", str(e))
                bad.add(s)
                continue
            if store is not None and store.done(s, k):
                out[(w.id, s)] = StageState("done", "", k)
                new_last.setdefault(str(w.id), {})[s] = k
                continue
            prev = prev_keys.get(s)
            if prev == k or not prev or store is None or not store.done(s, prev):
                prev = None
            tmp = _tmp_manifest(store, s, k)
            status = tmp.get("status")
            if status == "running" and k in live:
                out[(w.id, s)] = StageState("running", "", k, prev)
            elif status == "failed":
                out[(w.id, s)] = StageState("failed", str(tmp.get("error") or ""), k, prev)
                bad.add(s)
            elif status in ("interrupted", "running"):
                out[(w.id, s)] = StageState("interrupted", REASON_INTERRUPTED, k, prev)
            elif any(a in bad for a in keys.ancestors(w, s)):
                out[(w.id, s)] = StageState("skipped", REASON_UPSTREAM_FAILED, k, prev)
                bad.add(s)
            elif prev:
                out[(w.id, s)] = StageState("stale", _why_stale(keys, store, w, s, prev), k, prev)
            else:
                out[(w.id, s)] = StageState("pending", REASON_NEVER, k)
    if update and root is not None and new_last != last_keys and Path(root).is_dir():
        p2.write_state(root, {**last, "last_keys": new_last})
    return out


def _tmp_manifest(store: Store | None, stage: str, key: str) -> dict:
    if store is None:
        return {}
    try:
        return json.loads((store.tmp_dir(stage, key) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _tree_bytes(path: Path) -> int:
    total = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    total += _tree_bytes(Path(e.path))
                else:
                    try:
                        total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
    except OSError:
        pass
    return total


_HEX16 = re.compile(r"^[0-9a-f]{16}$")


def cache_usage(p: Project2) -> CacheUsage:
    """What the store and the scratch folder hold, and what no workflow needs: results whose key no workflow has now
    or had last, the ``.tmp`` folders of unfinished jobs (``resumable`` when a workflow has that key now), ``_trash``
    and scratch caches. Walks the folders: worker thread."""
    keys, store = _keys(p), _store(p)
    if store is None:
        return CacheUsage(0)
    now: set[tuple[str, str]] = set()
    for w in keys.project.workflows:
        for s in keys.chain(w):
            now.add((s, keys.key(w, s)))
    ref = set(now)
    for kv in p2.read_state(p.root)["last_keys"].values():
        ref |= {(s, k) for s, k in kv.items()}
    use = CacheUsage(0)
    if store.path.is_dir():
        for sd in sorted(store.path.iterdir()):
            if not sd.is_dir():
                continue
            if sd.name == "_trash":
                for t in sorted(sd.iterdir()):
                    use.unused.append(CacheItem("trash", t.name.split("-")[0], "", str(t), _tree_bytes(t)))
                continue
            if sd.name.startswith("_"):
                continue
            for d in sorted(sd.iterdir()):
                if not d.is_dir():
                    continue
                tmp = d.name.endswith(".tmp")
                k = d.name[:-4] if tmp else d.name
                size = _tree_bytes(d)
                if tmp:
                    use.unused.append(CacheItem("tmp", sd.name, k, str(d), size, resumable=(sd.name, k) in now))
                elif (sd.name, k) in ref:
                    use.used_bytes += size
                else:
                    use.unused.append(CacheItem("result", sd.name, k, str(d), size))
    scratch = Path(keys.project.scratch)
    own = _inside(scratch, Path(p.root))
    if scratch.is_dir():
        refkeys = {k for _, k in ref}
        for d in sorted(scratch.iterdir()):
            if not (d.is_dir() and _HEX16.match(d.name)):
                continue
            size = _tree_bytes(d)
            if d.name in refkeys:
                use.used_bytes += size
            elif own or any(_in_store(store, st.name, d.name) for st in d.iterdir() if st.is_dir()):
                use.unused.append(CacheItem("scratch", "", d.name, str(d), size))
    return use


def _in_store(store: Store, stage: str, key: str) -> bool:
    return (store.path / stage / key).exists() or (store.path / stage / (key + ".tmp")).exists()


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:                       # another drive
        return False


def clear_unused(p: Project2, items: Iterable[CacheItem] | None = None, include_resumable: bool = False) -> int:
    """Delete what ``cache_usage`` lists as unused (or the given ``items``, those still unused), after the client's
    confirmation; returns the bytes freed. Refuses during a run. The ``.tmp`` folders a workflow could resume from are
    kept unless ``include_resumable``. Only folders inside the store and the scratch folder are ever deleted."""
    root = Path(p.root)
    if _r2().running(root) is not None:
        raise RunInProgress("a run is in progress: stop it before clearing results")
    usage = cache_usage(p)
    wanted = {i.path for i in items} if items is not None else None
    store, scratch = _store(p).path, Path(_keys(p).project.scratch)
    freed = 0
    for it in usage.unused:
        if wanted is not None and it.path not in wanted:
            continue
        if it.resumable and not include_resumable:
            continue
        d = Path(it.path)
        if not (_inside(d, store) or _inside(d, scratch)) or d == store or d == scratch:
            continue
        try:
            shutil.rmtree(d)
            freed += it.bytes
        except OSError:
            pass
    _store(p).forget()
    return freed


# ---------------------------------------------------------------------------------------------- plans and runs

def _gpu(p: Project2, device: str | None = None) -> bool:
    d = (device or p.devices.get("main") or "auto").strip().lower()
    if d == "cpu":
        return False
    return True if d.startswith("cuda") else machine().cuda


def _plan_kw(p: Project2, device: str | None, history: dict | None) -> dict[str, Any]:
    keys = _keys(p)
    return {"keys": keys, "store": _store(p), "gpu": _gpu(p, device), "history": history}


def _stamp(p: Project2, plan: Plan) -> Plan:
    plan.rules = _ctx(p).spec
    return plan


def plan_all(p: Project2, device: str | None = None, history: dict | None = None) -> Plan:
    """Run all: every stage of every workflow, each result once (shared jobs), cheap workflows first."""
    r2 = _r2()
    return _stamp(p, r2.plan_all(_keys(p).project, **_plan_kw(p, device, history)))


def plan_workflows(p: Project2, workflows: Sequence[int | Workflow], device: str | None = None,
                   history: dict | None = None) -> Plan:
    """The plan computing every stage of these workflows (ids, or what-if Workflow objects)."""
    return _stamp(p, _r2().plan_workflows(_keys(p).project, list(workflows), **_plan_kw(p, device, history)))


def plan_to_block(p: Project2, wf: int | Workflow, block: str, device: str | None = None,
                  history: dict | None = None) -> Plan:
    """Run to here: the stages of a block in the workflow's chain (Selection: Registration) and what they need."""
    return _stamp(p, _r2().plan_to_block(_keys(p).project, wf, block, **_plan_kw(p, device, history)))


def cost(p: Project2, workflows: Sequence[int | Workflow] | None = None, device: str | None = None,
         history: dict | None = None) -> Cost:
    """What computing these workflows would take, on this machine: the plan's totals (jobs not done yet, each key
    once). Give what-if Workflow objects (``with_change``, a new workflow, a variant) to ask before changing."""
    plan = (plan_all(p, device, history) if workflows is None else plan_workflows(p, workflows, device, history))
    return Cost(plan.seconds, plan.measured, plan.disk_gb, len(plan), plan.keys, dict(plan.skipped))


def estimate(p: Project2, wf: int | Workflow, stage: str, device: str | None = None) -> Estimate:
    """Seconds (and disk) of one stage job in this workflow on this machine, and whether the figure was measured."""
    keys = _keys(p)
    return est.estimate(stage, est.facts(keys, wf, _store(p)), _gpu(p, device))


_MACHINE: list[MachineInfo] = []


def machine(force: bool = False) -> MachineInfo:
    """The GPUs (in CUDA's order, named as the engine names them), the CPU cores and the memory. Imports torch the
    first time: worker thread."""
    if _MACHINE and not force:
        return _MACHINE[0]
    gpus: list[Gpu] = []
    try:
        gpus = [Gpu(d.device, " ".join(w for w in d.name.split() if w != "GeForce"), d.memory_gb, d.beta)
                for d in devices.list_devices()]
    except Exception:  # noqa: BLE001 - no torch or no driver
        pass
    mem = 0.0
    try:
        import psutil
        mem = psutil.virtual_memory().total / 2 ** 30
    except Exception:  # noqa: BLE001
        pass
    _MACHINE[:] = [MachineInfo(gpus, os.cpu_count() or 0, mem)]
    return _MACHINE[0]


#: the run window says why (its translations: client/services/run.py, run.json)
STACK_ON_CPU = "the graphics card has {free} GB free and this stack needs about {need} GB: it is computed on the CPU"


def _stack_on_cpu(p: Project2, plan: Plan, device: str | None) -> dict[str, str]:
    """{key: why} of the Stack jobs of the plan that do not fit in the card's memory (``devices.stack_memory`` for the
    frame size and the drizzle scale), which go to the CPU."""
    dev = (device if device not in (None, "", "auto") else (p.devices.get("main") or "cuda:0"))
    if dev == "cpu":
        return {}
    try:
        free = devices.free_memory(dev) if devices.available(dev) else None
    except Exception:  # noqa: BLE001 - no torch or no driver
        free = None
    out: dict[str, str] = {}
    if free is None:
        return out
    keys, store = _keys(p), _store(p)
    for j in plan.jobs:
        if wfl.base(j.stage) != "stack":
            continue
        need = devices.stack_memory(est.facts(keys, j.run_wf, store).grid_mpx)
        if need * 2 ** 30 > free:
            out[j.key] = STACK_ON_CPU.format(free=f"{free / 2 ** 30:.1f}", need=f"{need:.1f}")
    return out


def start_run(p: Project2, plan: Plan, device: str | None = None, cpu_jobs: dict[str, str] | None = None) -> RunInfo:
    """Launch a plan in a detached worker process; returns at once. Follow it with ``read_events(p, info.offset,
    info.run_id)``. Raises ``RunInProgress`` when the project already has a live run. ``cpu_jobs``: {key: why} of the
    jobs that go to the CPU whatever the device (the card test); a Stack job that does not fit in the card's memory goes
    there too. The jobs left on the CPU have their duration estimated as such."""
    r2 = _r2()
    if r2.running(p.root) is not None:
        raise RunInProgress("a run is already in progress for this project")
    if plan.rules is None:
        plan.rules = _ctx(p).spec
    cpu = {**(cpu_jobs or {}), **_stack_on_cpu(p, plan, device)}
    if cpu:
        import dataclasses
        keys, store = _keys(p), _store(p)
        jobs = [dataclasses.replace(j, est_s=est.estimate(wfl.base(j.stage), est.facts(keys, j.run_wf, store),
                                                          False).seconds) if j.key in cpu else j for j in plan.jobs]
        plan = dataclasses.replace(plan, jobs=jobs)
    info = r2.launch(p, plan, device, cpu_jobs=cpu)
    refresh(p)
    return info


def running(p: Project2) -> RunInfo | None:
    """The live run of the project (to reattach), or None."""
    return _r2().running(p.root)


def cancel(p: Project2) -> None:
    """Ask the run to stop at its next check: the job in progress keeps its checkpoints for the next run."""
    _r2().cancel(p.root)


def read_events(p: Project2, offset: int = 0, run_id: str | None = None) -> tuple[list[dict], int]:
    """The events appended to ``logs/events.jsonl`` since byte ``offset`` (those of one run if ``run_id``), and the
    new offset."""
    return _r2().read_events(p.root, offset, run_id)


# ---------------------------------------------------------------------------------------------- outputs

def stage_outputs(p: Project2, wf: int | Workflow, stage: str, which: str = "current") -> StageOutputs | None:
    """The previews, metrics, parameters, origins, reasons, duration and date of a done result; ``which="previous"``
    for the result an Outdated stage had. None when there is none."""
    d = _dir(p, wf, stage, which)
    if d is None:
        return None
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    pv = d / "previews"
    previews = {f.stem: str(f) for f in sorted(pv.glob("*.jpg"))} if pv.is_dir() else {}
    return StageOutputs(stage, d.name, which, str(d), previews, m.get("metrics") or {}, m.get("params") or {},
                        m.get("origin") or {}, m.get("auto_reasons") or {}, m.get("duration"),
                        m.get("finished") or "", m)


def _reduce(a: Any, f: int) -> Any:
    import numpy as np
    if f <= 1 or a.ndim < 2:
        return a
    h, w = a.shape[-2] // f * f, a.shape[-1] // f * f
    b = a[..., :h, :w].astype(np.float32)
    return b.reshape(*b.shape[:-2], h // f, f, w // f, f).mean((-3, -1))


def stage_array(p: Project2, wf: int | Workflow, stage: str, name: str, scale: float = 1.0,
                which: str = "current") -> np.ndarray:
    """A linear array of a done result (``name`` with or without ``.npy``), read whole (never memory-mapped, so a
    re-run can replace the file); ``scale`` 1/2, 1/4... averages blocks over the last two axes."""
    import numpy as np
    d = _need(p, wf, stage, which)
    f = d / (name if name.endswith((".npy", ".npz")) else name + ".npy")
    if not f.is_file():
        raise NotDone(f"{stage} has no array {name!r}")
    return _reduce(np.load(f), max(1, int(round(1 / scale))))


_WCS: dict[str, tuple[float, Any]] = {}


def spectral_response(p: Project2, wf: int | Workflow, which: str = "current"):
    """The sensor's spectral response with its uncertainty (``core.spectral.SpectralResponse``), reconstructed from
    what the Color result measured on its colour stars and the Gaia XP spectra of the very Catalogs result it read:
    a diagnostic computed here in about a second, nothing stored. Raises ``NotDone`` without a Color result."""
    import numpy as np

    from .core import spectral
    d = _need(p, wf, "color", which)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    cat = d.parent.parent / "catalogs" / str((man.get("inputs") or {}).get("catalogs") or "")
    if not (cat / "gaia_xp.npz").is_file() or not (d / "color_stars.npz").is_file():
        raise NotDone(f"the colour stars of workflow {_wid(wf)} are not stored ({which})")
    with np.load(d / "color_stars.npz", allow_pickle=True) as z, np.load(cat / "gaia_xp.npz") as x:
        if not np.array_equal(z["source"], x["source"]):
            raise NotDone("the colour stars do not match the catalogue's")
        used = np.asarray(z["used"], bool) & np.isfinite(z["flux"]).all(1)
        if used.sum() < spectral.MIN_STARS:
            raise NotDone(f"{int(used.sum())} colour stars: too few to reconstruct the curves")
        return spectral.response(np.asarray(x["flux"], float)[used], np.asarray(x["lam"], float),
                                 np.asarray(z["flux"], float)[used])


def pixel_to_sky(p: Project2, wf: int | Workflow, x: float, y: float, which: str = "current") -> tuple[float, float]:
    """(RA, Dec) in degrees of a pixel of the stack grid (0-based), from the astrometric solution."""
    f = _need(p, wf, "astrometry", which) / "wcs.fits"
    mt = f.stat().st_mtime
    hit = _WCS.get(str(f))
    if hit is None or hit[0] != mt:
        from astropy.io import fits
        from astropy.wcs import WCS
        hit = _WCS[str(f)] = (mt, WCS(fits.getheader(f)))
    ra, dec = hit[1].pixel_to_world_values(x, y)
    return float(ra), float(dec)


def sky_objects(p: Project2, wf: int | Workflow, which: str = "current") -> list[SkyObject]:
    """The named deep-sky objects (galaxies, nebulae, clusters: Messier, NGC, IC, Sharpless) whose centre or outline
    falls in the field of the workflow's astrometric solution (the stack grid it was solved on), biggest first.
    ``SkyObject``: name ("M 42", "IC 1805", "Sh2-275"), common name ("Heart Nebula" or ""), kind, RA / Dec in degrees
    J2000, axes in arcminutes, position angle in degrees east of north, other designations. ``NotDone`` when Astrometry
    is not computed; ``CatalogueUnavailable`` (a plain message) when the catalogue shipped with the package is
    incomplete. Where the names come from: ``core.skyobjects.catalogue_source()``. The first call of a session reads
    the catalogue (a few tens of ms from its cache, about a second the first time): worker thread."""
    import warnings

    import numpy as np

    from .core import skyobjects as so
    d = _need(p, wf, "astrometry", which)
    f = d / "wcs.fits"
    if not f.is_file():
        raise NotDone(f"the astrometry result of workflow {_wid(wf)} has no solution ({which})")
    mt = f.stat().st_mtime
    hit = _WCS.get(str(f))
    if hit is None or hit[0] != mt:
        from astropy.io import fits
        from astropy.wcs import WCS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hit = _WCS[str(f)] = (mt, WCS(fits.getheader(f)))
    st = _upstream(p, wf, d, "stack", which)
    if st is None:
        raise NotDone(f"the stack the astrometry of workflow {_wid(wf)} was solved on is gone ({which})")
    meta = _json(st / "drz_meta.json") if (st / "drz_meta.json").is_file() else {}
    if meta.get("width") and meta.get("height"):
        w, h = int(meta["width"]), int(meta["height"])
    else:
        h, w = np.load(st / "drz_full.npy", mmap_mode="r").shape[-2:]
    return so.objects_in_field(hit[1], w, h)


# ---------------------------------------------------------------------------------------------- diagnostics of the basics

_TITLES = {"calibrate": "Calibration", "register": "Registration", "stack": "Stacking", "color": "Colour calibration"}
_DIAG: dict[tuple, Any] = {}             # results computed again from a stack (color_fit of an older result)


def _missing(what: str, stage: str) -> str:
    return f"{what}: missing: recompute {_TITLES.get(stage, stage)} to get it"


def _manifest_of(d: Path) -> dict:
    try:
        return json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _upstream(p: Project2, wf: int | Workflow, d: Path, up: str, which: str) -> Path | None:
    """The done result of stage ``up`` that the result in ``d`` was made from (the upstream keys its manifest names,
    followed up the chain); the workflow's result of ``up`` when the manifests do not say."""
    store = _store(p)
    if store is not None:
        seen: set[tuple[str, str]] = set()
        todo = [d]
        while todo:
            m = _manifest_of(todo.pop())
            ups = dict(m.get("inputs") or {})
            ups.update((m.get("key_parts") or {}).get("u") or {})
            for s, k in ups.items():
                if not isinstance(k, str) or (s, k) in seen:
                    continue
                seen.add((s, k))
                try:
                    dd = store.dir(s, k)
                except ValueError:                       # a v1 run id, not a key
                    continue
                if not store.done(s, k):
                    continue
                if s == up:
                    return dd
                todo.append(dd)
    return _dir(p, wf, up, which) or _dir(p, wf, up)


def _json(f: Path) -> Any:
    return json.loads(f.read_text(encoding="utf-8"))


def _norm(f: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(f)))


def _num(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def calibration_frames(p: Project2, session_id: int, wf: int | Workflow | None = None, which: str = "current", *,
                       require_done: bool = True,
                       progress: Callable[[float, str], None] | None = None) -> CalibrationReport:
    """The calibration of one session: the masters of the calibration result (``wf``'s, else the current workflow's,
    else any workflow's that is done; they are made from the frames of all the sessions), the measures and the quality
    flag of each of the session's darks, flats, bias and flat darks (measured on demand on the raw files, kept in
    memory: the first call reads every frame) and its lights with their calibrated frame. ``NotDone`` when no
    calibration result is done (``require_done=False``: the frames are measured anyway, without masters). Worker
    thread (it reads the raw frames)."""
    import numpy as np

    from .core import diagnostics as dg
    from .io.frames import read_info
    say = progress or (lambda f, m: None)
    keys = _keys(p)
    proj = keys.project
    session = proj.session(session_id)
    d: Path | None = None
    if wf is not None:
        d = _dir(p, wf, "calibrate", which)
    else:
        for wid in dict.fromkeys(([proj.current] if proj.current is not None else []) + [w.id for w in proj.workflows]):
            d = _dir(p, wid, "calibrate", which)
            if d is not None:
                wf = wid
                break
    if d is None and require_done:
        who = f"workflow {_wid(wf)}" if wf is not None else "any workflow"
        raise NotDone(f"the calibrate result of {who} is not computed ({which})")
    rep = CalibrationReport(session_id, stage_dir=str(d or ""))
    inv_frames: dict[str, list[dict]] = {}
    metrics: dict[str, Any] = {}
    if d is not None:
        inv = _upstream(p, wf, d, "inventory", which) if wf is not None else None
        if inv is not None and (inv / "frames.json").is_file():
            inv_frames = _json(inv / "frames.json")
        metrics = _manifest_of(d).get("metrics") or {}
        for kind, name in (("dark", "master_dark.npy"), ("flat", "master_flat.npy"), ("bias", "master_bias.npy"),
                           ("flat_dark", "master_flat_dark.npy"), ("bad_pixels", "badmask.npy")):
            if (d / name).is_file():
                rep.masters[kind] = str(d / name)
        union = proj.inputs()
        for kind, frames_kind, what in (("bias", "bias", "master bias"), ("flat_dark", "flat_darks", "master flat dark")):
            had = inv_frames.get(frames_kind) if inv_frames else getattr(union, frames_kind)
            if had and kind not in rep.masters:
                rep.missing.append(_missing(what, "calibrate"))
    files = keys.scan.session_frames(proj, session)

    # ---------------- the calibration frames, measured on the raws
    jobs = [(kind, Path(f)) for kind in ("darks", "flats", "bias", "flat_darks") for f, _ in files[kind]]
    firsts: dict[str, Any] = {}
    for kind, f in jobs:
        if kind not in firsts:
            firsts[kind] = read_info(f)

    def measure(job: tuple[str, Path]) -> dict[str, Any]:
        kind, f = job
        info = firsts[kind] if Path(firsts[kind].path) == f else read_info(f, like=firsts[kind])
        st = dg.raw_stats(f, info.white)
        t = info.temperature if info.temperature is not None else (
            dg.raw_temperature(f) if info.kind == "raw" else None)
        st.update(exposure=info.exposure, iso=info.iso, temperature=t, time=info.time, white=info.white,
                  black=info.black)
        return st

    stats: list[dict[str, Any]] = []
    if jobs:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(4, len(jobs))) as ex:
            for i, st in enumerate(ex.map(measure, jobs)):
                stats.append(st)
                say((i + 1) / len(jobs), f"measuring the calibration frames {i + 1}/{len(jobs)}")
    by_kind: dict[str, list[int]] = {}
    for i, (kind, _) in enumerate(jobs):
        by_kind.setdefault(kind, []).append(i)

    def level(kind: str) -> float | None:
        v = [stats[i]["median"] for i in by_kind.get(kind, [])]
        return float(np.median(v)) if v else None

    zero = level("flat_darks")
    zero = zero if zero is not None else level("bias")
    zero = zero if zero is not None else _num(metrics.get("dark_level"))
    zero = zero if zero is not None else level("darks")
    for i in by_kind.get("flats", []):
        st = stats[i]
        z = zero if zero is not None else float(st["black"])
        st.update(zero=z, **dg.flat_measures(st, z, float(st["white"])))
    for kind, idx in by_kind.items():
        for i, (flag, reason, checks) in zip(idx, dg.flag_frames(kind, [stats[i] for i in idx])):
            f = jobs[i][1]
            rep.frames.append(CalFrame(kind, str(f), f.name, stats[i], flag, reason, checks))

    # ---------------- the lights, as the calibration made them
    lights_inv = inv_frames.get("lights") or []
    light_exp: list[float] = []
    if d is not None and lights_inv:
        pos = {_norm(f["path"]): i for i, f in enumerate(lights_inv)}
        fs = np.load(d / "frame_stats.npy") if (d / "frame_stats.npy").is_file() else None
        cache = _json(d / "cache.json") if (d / "cache.json").is_file() else {}
        cache_ok = bool(cache.get("path")) and Path(cache["path"]).is_file()
        if cache and not cache_ok:
            rep.notes.append("The calibrated light frames are no longer on the scratch disk: the next run that needs "
                             "them makes them again.")
        rows = []
        for f, _ in files["lights"]:
            i = pos.get(_norm(f))
            if i is None:
                continue
            if lights_inv[i].get("exposure"):
                light_exp.append(float(lights_inv[i]["exposure"]))
            st: dict[str, Any] = {"time": lights_inv[i].get("time"), "exposure": lights_inv[i].get("exposure"),
                                  "iso": lights_inv[i].get("iso"), "temperature": lights_inv[i].get("temperature")}
            if fs is not None and i < len(fs):
                st.update(median=float(np.median(fs[i, :, 0])), noise=float(np.median(fs[i, :, 1])),
                          planes=[float(v) for v in fs[i, :, 0]], noise_planes=[float(v) for v in fs[i, :, 1]])
            if cache:
                st["shape"] = [int(cache["n"]), int(cache["height"]), int(cache["width"])]
                st["cache_scale"] = cache.get("scale")
            rows.append(CalFrame("lights", str(f), Path(f).name, st, calibrated=cache["path"] if cache_ok else "",
                                 index=i))
        for r, (flag, reason, checks) in zip(rows, dg.flag_frames("lights", [r.stats for r in rows])):
            r.flag, r.reason, r.checks = flag, reason, checks
        rep.frames += rows

    # ---------------- notes
    if len(proj.sessions) > 1 and d is not None:
        rep.notes.append(f"The masters are made from the frames of all {len(proj.sessions)} sessions together (one "
                         "master dark, one master flat).")
    src = inv_frames or {k: getattr(proj.inputs(), k) for k in KINDS}
    for kind, text in (("darks", "No darks: the master bias (or the camera's black level) stands for the master "
                                 "dark; hot pixels are left to the bad pixel map."),
                       ("flats", "No flats: vignetting and dust are not corrected.")):
        if d is not None and not src.get(kind):
            rep.notes.append(text)
        elif src.get(kind) and not files[kind] and d is not None:
            rep.notes.append(f"This session has no {kind} of its own: the master comes from the other sessions.")
    dark_exp =[stats[i]["exposure"] for i in by_kind.get("darks", []) if stats[i].get("exposure")]
    if light_exp and dark_exp:
        le, de = float(np.median(light_exp)), float(np.median(dark_exp))
        if abs(le - de) > 0.01 * max(le, de):
            rep.notes.append(f"The darks are {de:g} s exposures and the lights {le:g} s: the master dark does not "
                             "match the dark current of the lights.")
    return rep


def calibrated_light(frame: CalFrame) -> np.ndarray:
    """The calibrated mosaic (H, W) float32 of a light of ``calibration_frames``, read from its cache: dark subtracted,
    flat divided, in the cache's ADU (``stats["cache_scale"]``); NaN where a pixel is excluded (hot, cold, dust),
    above 60000 where saturated. Read whole, the file closed after. ``NotDone`` when the cache is gone. Worker
    thread."""
    import numpy as np
    if frame.kind != "lights" or not frame.calibrated or not Path(frame.calibrated).is_file() or frame.index is None:
        raise NotDone(f"no calibrated frame of {frame.name} in the cache")
    n, h, w = frame.stats["shape"]
    mm = np.memmap(frame.calibrated, dtype=np.float16, mode="r", shape=(n, h, w))
    try:
        return np.asarray(mm[frame.index], np.float32)
    finally:
        del mm


def registration_table(p: Project2, wf: int | Workflow, which: str = "current") -> RegTable:
    """One row per light frame as registration aligned it: dx, dy (px), rotation (degrees), scale, stars matched and
    detected, FWHM, roundness, whether it was aligned (``RegRow``). ``NotDone`` when Registration is not computed.
    Worker thread (it reads reg.npz and the frame list)."""
    import numpy as np

    from .core import diagnostics as dg
    d = _need(p, wf, "register", which)
    with np.load(d / "reg.npz") as z:
        reg = {k: z[k] for k in z.files}
    n = len(reg["fwhm"])
    inv = _upstream(p, wf, d, "inventory", which)
    lights = (_json(inv / "frames.json").get("lights") or []) if inv is not None else []
    im = (_manifest_of(inv).get("metrics") or {}) if inv is not None else {}
    width, height = _num(im.get("width")), _num(im.get("height"))
    if (width is None or height is None) and lights:
        width, height = _num(lights[0].get("width")), _num(lights[0].get("height"))
    rms = np.asarray(reg["rms"], float)
    aligned = np.isfinite(rms)
    notes = []
    if width is None or height is None:
        notes.append("The sensor size is unknown (no inventory result): no offsets, rotation or scale.")
        dx = dy = rot = sc = np.full(n, np.nan)
    else:
        dx, dy, rot, sc = dg.frame_geometry(reg["coef"], width, height, aligned)
    rnd = reg.get("roundness")
    ref = int(reg["ref"]) if "ref" in reg else -1
    def opt(name, i):                    # register's Moffat arrays (absent from a result made before them)
        a = reg.get(name)
        return None if a is None or not np.isfinite(a[i]) else float(a[i])

    rows = []
    for i in range(n):
        f = lights[i] if i < len(lights) else {}
        r = None if rnd is None or not np.isfinite(rnd[i]) else float(rnd[i])
        row = RegRow(Path(f.get("path", f"frame {i + 1}")).name, f.get("time"), float(dx[i]), float(dy[i]),
                     float(rot[i]), float(sc[i]), int(reg["nmatch"][i]) if aligned[i] else 0,
                     int(reg["nstars"][i]) if "nstars" in reg else 0, float(reg["fwhm"][i]), r, bool(aligned[i]),
                     i, f.get("path", ""), float(rms[i]), float(reg["transp"][i]), i == ref,
                     "" if aligned[i] else "not aligned")
        row.fwhm_moffat, row.roundness_moffat, row.beta_moffat = (opt(k, i) for k in ("fwhm_moffat",
                                                                                       "roundness_moffat",
                                                                                       "beta_moffat"))
        rows.append(row)
    missing = [] if "roundness" in reg else [_missing("roundness", "register")]
    if "nstars" not in reg:
        missing.append(_missing("stars detected", "register"))
    return RegTable(rows, missing, notes)


def stack_diagnostics(p: Project2, wf: int | Workflow, which: str = "current") -> StackDiag:
    """The rejection maps (low and high), the weight and normalisation of each frame, the frames used and the noise of
    each channel of a done stack (``StackDiag``). For a stack made before it kept them the maps and the normalisation
    are missing, and the noise is measured on its half-stacks (the first call reads them). ``NotDone`` when Stacking
    is not computed. Worker thread."""
    import numpy as np

    from .core import diagnostics as dg
    from .stages.stack import DIAG, REJECTION
    d = _need(p, wf, "stack", which)
    m = _manifest_of(d)
    sf = _json(d / "frames.json")
    names = [Path(f["path"]).name for f in sf]
    out = StackDiag(rejected_pct=_num((m.get("metrics") or {}).get("rejected_pct")))
    out.used = [names[i] for i, f in enumerate(sf) if float(f.get("weight") or 0) > 0]
    out.excluded = {names[i]: str(f.get("excluded") or "") for i, f in enumerate(sf)
                    if float(f.get("weight") or 0) <= 0}
    meta = _json(d / "drz_meta.json") if (d / "drz_meta.json").is_file() else {}
    if meta.get("width") and meta.get("height"):
        out.grid = (int(meta["width"]), int(meta["height"]))
    diag = _json(d / DIAG) if (d / DIAG).is_file() else None
    by_index: dict[int, dict] = {}
    if diag is not None:
        out.frames = list(diag.get("frames") or [])
        by_index = {int(f["index"]): f for f in out.frames}
        out.noise = {k: (float(v) if v is not None else float("nan")) for k, v in (diag.get("noise") or {}).items()}
        on = sum(int(f.get("rejected_low", 0)) + int(f.get("rejected_high", 0)) for f in out.frames)
        off = sum(int(f.get("rejected_outside", 0)) for f in out.frames)
        sensor = meta.get("sensor") or []
        if len(sensor) == 2 and out.frames:
            out.rejected_on_grid_pct = 100.0 * on / (len(out.frames) * float(sensor[0]) * float(sensor[1]))
        if off > on:
            out.notes.append(f"{100.0 * off / (on + off):.0f} % of the rejected samples land outside the stack (the "
                             "margins the drift uncovers, where nothing is predicted): the stage's 'rejected' figure "
                             f"counts them; on the stack, {out.rejected_on_grid_pct or 0:.2f} % of the samples were "
                             "rejected.")
    else:
        out.missing.append(_missing("normalisation of each frame", "stack"))
        try:
            hb = int((m.get("params") or {}).get("half_block") or 20)
            out.noise = dg.stack_noise(d, [float(f.get("weight") or 0) for f in sf], hb)
            out.notes.append("The noise was measured on the half-stacks: this stack was made before it kept it.")
        except (OSError, ValueError):
            out.missing.append(_missing("noise", "stack"))
    for i, f in enumerate(sf):
        fr = by_index.get(i)
        out.weights.append((names[i], float(f.get("weight") or 0),
                            tuple(float(v) for v in fr["offset"]) if fr else None,
                            tuple(float(v) for v in fr["scale"]) if fr else None))
    if (d / REJECTION).is_file():
        with np.load(d / REJECTION) as z:
            out.rejection_low, out.rejection_high = z["low"], z["high"]
            out.rejection_factor = int(z["factor"])
            if "untested" in z.files:
                out.rejection_untested = z["untested"]
    else:
        out.missing.append(_missing("rejection maps", "stack"))
    _stack_details(out, d, m, sf, names, diag)
    return out


#: the coverage maps of the stacks made before the stack kept them, computed again from drz_weight.npy: one at a
#: time (a stack's weight map is large); (folder, its mtime) -> the record
_COVERAGE: dict[tuple, dict] = {}


def _stack_details(out: StackDiag, d: Path, m: dict, sf: list, names: list[str], diag: dict | None) -> None:
    """StackDiag's details: the switches, the rejection's rates, the coverage per colour, the weights per colour, the
    reasons of the frames left out, the selection, the normalisation and the colour alignment. A stack made before
    the stack kept them gets the same fields from what it has."""
    import numpy as np

    from .stages.stack import COVERAGE, THRESHOLD_CODES, reduce_map
    metrics = m.get("metrics") or {}
    diag = diag or {}
    out.switches = dict(diag.get("switches") or {})
    rej = dict(diag.get("rejection") or {})
    summ = rej.get("summary") or {}
    out.rejection = {"method": rej.get("method", "own"), "rejected_pct": _num(metrics.get("rejected_pct")),
                     "untested_pct": _num(metrics.get("untested_pct")), "reference_pct": [0.1, 0.5]}
    for k in ("low_pct", "high_pct", "exempt_pct", "z_spread_median"):
        if k in summ:
            out.rejection[k] = summ[k]
    for k in ("classic", "reason_template"):           # the Classic method: the rejection its rule chose, and why
        if rej.get(k):
            out.rejection[k] = rej[k]
    if rej.get("prediction"):
        out.rejection["prediction"] = rej["prediction"]
    # ---- weights per colour, the reasons of the frames left out
    for i, f in enumerate(sf):
        w = float(f.get("weight") or 0)
        out.weights_rgb[names[i]] = tuple(float(v) for v in (f.get("weights_rgb") or [w, w, w]))
        if w <= 0:
            if "exclusions" in f:
                out.exclusions[names[i]] = [dict(r) for r in f["exclusions"]]
            elif f.get("excluded"):
                r = str(f["excluded"])
                out.exclusions[names[i]] = [{"code": THRESHOLD_CODES.get(r, "selection"), "template": r, "args": None}]
    out.selection = diag.get("selection")
    out.normalisation = diag.get("normalisation")
    ca = diag.get("channel_alignment")
    if ca is None and metrics.get("channel_offsets") is not None:
        ca = {**metrics["channel_offsets"], "mode": "night"}
    out.channel_alignment = ca
    # ---- the coverage per colour: kept by the stack (coverage.npz), else from its weight map
    cov: dict[str, Any] = {}
    if (d / COVERAGE).is_file():
        with np.load(d / COVERAGE) as z:
            cov = {"factor": int(z["factor"]), "maps": z["measured"],
                   "variation": {c: _num(v) for c, v in zip("RGB", z["variation"])},
                   "estimate_map": z["estimate_map"]}
        rec = diag.get("coverage") or {}
        cov["estimate"] = {k: rec.get(k) for k in ("estimate_cv", "estimate_cv_max", "scale", "pixfrac", "limit")}
        if rec.get("rule"):
            cov["rule"] = rec["rule"]
    elif (d / "drz_weight.npy").is_file():
        ck = (str(d), (d / "drz_weight.npy").stat().st_mtime_ns)
        cov = _COVERAGE.get(ck)
        if cov is None:
            from .core.coverage import map_variation
            w = np.load(d / "drz_weight.npy", mmap_mode="r")
            f = out.rejection_factor if out.rejection_low is not None else max(1, -(-max(w.shape[1:]) // 4096))
            cov = {"factor": int(f), "maps": reduce_map(w, f),
                   "variation": {c: _num(v) for c, v in map_variation(np.asarray(w)).items()}}
            del w
            _COVERAGE.clear()
            _COVERAGE[ck] = cov
    out.coverage = cov                              # {} without a weight map (a stack made before it kept one)


def color_fit(p: Project2, wf: int | Workflow, which: str = "current") -> ColorFit:
    """The photometric colour fit: every catalogue star with its measured, catalogue and predicted colour, residual
    and whether the fit used it (and why not), the fitted matrix and the fit statistics (``ColorFit``). For a result
    made before the colour stage kept its stars, they are measured again on its stack (the same code: the same stars;
    the first call reads the stack). ``NotDone`` when Colour calibration is not computed. Worker thread."""
    import numpy as np

    from .core import diagnostics as dg
    from .stages.color import STAR_REASONS, STARS, ColorParams
    d = _need(p, wf, "color", which)
    m = _manifest_of(d)
    metrics, params = m.get("metrics") or {}, m.get("params") or {}
    out = ColorFit(matrix=np.load(d / "M.npy") if (d / "M.npy").is_file() else None)
    aperture = _num(params.get("aperture")) or ColorParams().aperture
    white_temp = _num(params.get("white_temp")) or ColorParams().white_temp
    source_kind = str(metrics.get("response_source") or "")
    cat_dir = _upstream(p, wf, d, "catalogs", which)
    computed = "stage"
    if (d / STARS).is_file():
        with np.load(d / STARS) as z:
            t = {k: z[k] for k in z.files}
        source_kind = str(t.get("source_kind", source_kind))
    else:                                            # a result made before the stage kept its stars: run it again
        out.missing.append(_missing("the fit of each star", "color"))
        out.stats = {"response_source": source_kind, "aperture": aperture, "white_temp": white_temp}
        return out
    bprp = dg.bp_rp(cat_dir, t["source"])
    flux, cam_pred = np.asarray(t["flux"], float), np.asarray(t["cam_pred"], float)
    from_fit = source_kind == "fit"
    for i in range(len(t["x"])):
        fl = flux[i]
        meas = tuple(float(v) for v in fl / fl.sum()) if np.isfinite(fl).all() and fl.sum() > 0 else None
        cp = cam_pred[i]
        used = bool(t["used"][i])
        why = str(t["reason"][i])
        out.stars.append(StarFit(
            float(t["x"][i]), float(t["y"][i]), meas, (float(t["cat_xy"][i, 0]), float(t["cat_xy"][i, 1])),
            (float(t["pred_xy"][i, 0]), float(t["pred_xy"][i, 1])) if used else None,
            float(t["resid"][i]) if used else None, used, "" if used else STAR_REASONS.get(why, why),
            int(t["source"][i]), float(t["g"][i]), float(bprp[i]) if np.isfinite(bprp[i]) else None,
            tuple(float(v) for v in fl) if np.isfinite(fl).all() else None,
            tuple(float(v) for v in cp / cp.sum()) if np.isfinite(cp).all() and cp.sum() > 0 else None,
            ",".join(c for c, k in zip("RGB", t["resp_keep"][i]) if not k) if used and from_fit else ""))
    res = np.array([s.residual for s in out.stars if s.used], float)
    rejected: dict[str, int] = {}
    for s in out.stars:
        if not s.used:
            rejected[s.reason] = rejected.get(s.reason, 0) + 1
    out.stats = {"stars": len(out.stars), "used": int(len(res)), "rejected": rejected,
                 "chromaticity_residual": float(np.median(res)) if len(res) else None,
                 "residual_p90": float(np.percentile(res, 90)) if len(res) else None,
                 "response_outliers": sum(1 for s in out.stars if s.outlier),
                 "response_source": source_kind, "fit_residual": metrics.get("fit_residual"),
                 "fit_agreement_pct": metrics.get("fit_agreement_pct"),
                 "profile_agreement_pct": metrics.get("profile_agreement_pct"),
                 "matrix_xyz": np.asarray(t["M_xyz"], float).tolist(), "aperture": aperture,
                 "white_temp": white_temp, "computed": computed}
    return out


# ---------------------------------------------------------------------------------------------- describe

@dataclass
class _Resolved:
    values: dict[str, Any]
    origin: dict[str, str]
    suggested: dict[str, Any]
    known: dict[str, bool]
    autos: dict[str, Any]
    auto_ok: bool


def _autos(p: Project2, keys: Keys, w: Workflow, stage: str) -> tuple[dict[str, Any], bool]:
    """(the stage's automatic values, whether they could be computed): needs the stages it reads done. Computed on
    the CPU in a throw-away folder, so that nothing is written."""
    store = _store(p)
    if store is None:
        return {}, False
    anc = keys.ancestors(w, stage)
    if not all(store.done(a, keys.key(w, a)) for a in anc):
        return {}, False
    st = keys.rules.chains.stages[wfl.base(stage)]
    c = _ctx(p)
    ck = (stage, keys.key(w, stage))
    if ck in c.autos:
        return c.autos[ck], True
    from .core.stage import Context
    try:
        with tempfile.TemporaryDirectory(prefix="astro_auto_", ignore_cleanup_errors=True) as tmp:
            sp = StoreProject(keys, w, stage, own_dir=tmp)
            auto = st.auto(Context(sp, st, _NullJournal(), "cpu"))
    except Exception:  # noqa: BLE001 - the upstream files are not readable: no suggestion
        return {}, False
    c.autos[ck] = auto
    return auto, True


def _resolve(p: Project2, wf: int | Workflow, stage: str) -> _Resolved:
    from .core.params import resolve
    keys = _keys(p)
    w = keys.project.workflow(wf)
    if stage not in keys.chain(w):
        raise ValueError(f"workflow {w.name!r} has no {stage!r} stage")
    model = keys.rules.params(wfl.base(stage))
    autos, ok = _autos(p, keys, w, stage)
    given = keys.overrides(w, stage)
    try:
        params, origin = resolve(model, autos, given)
        sug, _ = resolve(model, autos, {})
    except ValueError:                                 # an automatic value the settings refuse: as if unknown
        autos, ok = {}, False
        params, origin = resolve(model, autos, given)
        sug, _ = resolve(model, autos, {})
    auto_f = keys.rules.auto_fields(stage)
    known = {n: (n in autos) or ok or (n not in auto_f) for n in model.model_fields}
    return _Resolved(params.model_dump(mode="json"), origin, sug.model_dump(mode="json"), known, autos, ok)


def describe(p: Project2, wf: int | Workflow, stage: str) -> list[ParamInfo]:
    """The settings of a stage in a workflow, for the panel: label, help, bounds, the effective value, the suggestion
    and whether it is known, where the value comes from, the engine's reason (template and arguments, as the Auto
    object has them) and any drift from the done result. Worker thread (it may read the upstream results)."""
    keys, store = _keys(p), _store(p)
    w = keys.project.workflow(wf)
    r = _resolve(p, w, stage)
    b = wfl.base(stage)
    model = keys.rules.params(b)
    locked = set(keys.rules.implied(w.options, b)) | ({"exclude"} if keys.rules.injects_exclude(b) else set())
    used: dict[str, Any] = {}
    if store is not None and store.done(stage, keys.key(w, stage)):
        used = store.manifest(stage, keys.key(w, stage)).get("params") or {}
    skip = keys.rules.non_key(b)
    out = []
    for d in model.describe():
        n = d["name"]
        a = r.autos.get(n) if r.origin[n] == "auto" else None
        drift = bool(used) and n in used and n not in skip and used[n] != r.values[n]
        english = getattr(a, "reason", "")
        adv = r.autos.get(n)
        adv = adv if getattr(adv, "advice", False) else None
        out.append(ParamInfo(
            n, d.get("label", n), d.get("help", ""), d.get("tip", ""), d["type"], d["default"], r.values[n],
            r.suggested[n], r.known[n], r.origin[n],
            reason=getattr(a, "template", english), reason_args=getattr(a, "args", None), reason_en=english,
            level=d.get("level", "basic"), unit=d.get("unit", ""), ge=d.get("ge"), le=d.get("le"), gt=d.get("gt"),
            lt=d.get("lt"), step=d.get("step"), choices=d.get("choices"), locked=n in locked, drift=drift,
            result_value=used.get(n) if drift else None,
            advice=None if adv is None else adv.value, advice_reason="" if adv is None else adv.template,
            advice_args=None if adv is None else adv.args, advice_reason_en="" if adv is None else adv.reason))
    return out


# ---------------------------------------------------------------------------------------------- rendering and export

def sliders_of(p: Project2, wf: int | Workflow, part: str | None = None) -> list[SliderInfo]:
    """The Post sliders of a workflow's layer and which ones it offers (the reason when it does not). ``part``: None
    for the Output block's image, else one of the Stars block's: 'starless', 'stars', 'combined'."""
    from .render.sliders import SLIDERS, availability
    w = _keys(p).project.workflow(wf)
    kind = part or ("stacking" if w.end == "stacking" else "simple")
    why = availability(kind, bool(w.options.get("denoise")))
    return [SliderInfo(s.key, s.name, s.group, s.ends, not why[s.key], why[s.key]) for s in SLIDERS]


_SERVICES: dict[str, Any] = {}


def layer_service(device: str = "cpu") -> Any:
    """The shared ``LayerService`` of a device (engines kept, least recently used dropped); the render worker may
    pass its own to ``render_layer``."""
    from .render.layers import LayerService
    if device not in _SERVICES:
        _SERVICES[device] = LayerService(device)
    return _SERVICES[device]


def _output(p: Project2, wf: int | Workflow, part: str | None = None) -> tuple[str, Path]:
    """(stage, folder) of a layer's images: the Output block's, or the Stars block's for its parts."""
    keys = _keys(p)
    w = keys.project.workflow(wf)
    block = "output" if part is None else "stars"
    outs = [s for s in keys.chain(w) if keys.rules.chains.block_of(s) == block]
    if not outs:
        raise NotDone(f"workflow {w.name!r} has no {'Output' if part is None else 'Stars'} block")
    return outs[-1], _need(p, w, outs[-1])


def _base_of(p: Project2, base: Any) -> Any:
    return p.render_bases[base] if isinstance(base, str) and base in p.render_bases else base


def render_layer(p: Project2, wf: int | Workflow, base: Any = None, sliders: dict[str, float] | None = None,
                 scale: float = 0.25, region: tuple[int, int, int, int] | None = None, part: str | None = None,
                 blend: str | None = None, *, service: Any = None, device: str = "cpu") -> LayerImage:
    """A workflow's layer on the whole stack grid (or the region (x0, x1, y0, y1) of it at full resolution), uncropped,
    with the useful frame in ``crop``. ``base``: a name of ``render_bases``, settings or None (the suggestion);
    ``sliders``: positions 0-4 (2 = the base); ``part``: None (the Output block's image), or the Stars block's
    'starless', 'stars' or 'combined' (``blend``: how the stars go back, render/stars.BLENDS). ``NotDone`` when the
    stage is not computed."""
    _, d = _output(p, wf, part)
    return (service or layer_service(device)).render(d, _base_of(p, base), sliders, scale, region, part, blend)


def post_masks(p: Project2, wf: int | Workflow, scale: float = 0.25, region: tuple[int, int, int, int] | None = None,
               *, service: Any = None, device: str = "cpu") -> dict[str, Any]:
    """Post's data masks of a layer: ``halpha``, float32 in [0, 1]."""
    _, d = _output(p, wf)
    return (service or layer_service(device)).masks(d, scale, region)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "image"


def export_image(p: Project2, rgb: np.ndarray, meta: dict[str, Any] | None = None,
                 options: dict[str, Any] | Any = None, progress: Callable[[float, str], None] | None = None) -> list[Path]:
    """Write a composed image (3, H, W), display sRGB in [0, 1]: TIFF 16 / 32f, PNG 16, JPEG, FITS 32f with the WCS, at
    full size or reduced, in a v1 orientation, with a history file. ``options``: ``ExportOptions`` or its fields.
    ``meta``: ``wf`` (the workflow whose solution goes in the FITS), ``origin`` (grid pixel of ``rgb[:, 0, 0]``),
    ``history`` (dict), ``out_dir`` (default ``export/<name>/``). Returns the files."""
    from .render.layers import ExportOptions, export_rgb
    meta = dict(meta or {})
    o = options if isinstance(options, ExportOptions) else ExportOptions(**(options or {}))
    out_dir = Path(meta.get("out_dir") or Path(p.export) / _slug(o.name))
    wcs = meta.get("wcs")
    if wcs is None and meta.get("wf") is not None:
        f = _dir(p, meta["wf"], "astrometry")
        wcs = f / "wcs.fits" if f is not None and (f / "wcs.fits").is_file() else None
    return export_rgb(rgb, out_dir, o, wcs, tuple(meta.get("origin") or (0, 0)), meta.get("history"),
                      progress or (lambda f, m: None))


def export_workflow(p: Project2, wf: int | Workflow, base: Any = None, sliders: dict[str, float] | None = None,
                    options: dict[str, Any] | Any = None, progress: Callable[[float, str], None] | None = None,
                    *, service: Any = None, device: str = "cpu") -> list[Path]:
    """Headless export of one workflow's layer at full resolution, cropped to its useful frame, without Post's
    composition. The folder is ``export/<name>/`` (name: the options' name, else the workflow's)."""
    keys = _keys(p)
    w = keys.project.workflow(wf)
    opts = dict(options or {}) if not hasattr(options, "model_dump") else options.model_dump()
    opts.setdefault("name", _slug(w.name))
    lay = render_layer(p, w, base, sliders, 1.0, service=service, device=device)
    x0, x1, y0, y1 = lay.crop
    hist = {"project": p.name, "workflow": {"id": w.id, "name": w.name, "options": w.options},
            "base": base, "sliders": sliders or {}, "created": datetime.now().isoformat(timespec="seconds"),
            "keys": keys.keys(w), "render": lay.info}
    return export_image(p, lay.rgb[:, y0:y1, x0:x1], {"wf": w.id, "origin": (x0, y0), "history": hist}, opts, progress)


# ---------------------------------------------------------------------------------------------- lazy re-exports

def __getattr__(name: str) -> Any:
    """``Plan``, ``Job``, ``RunInfo`` (runner2), ``LayerImage`` and ``ExportOptions`` (render/layers), ``ImportReport``,
    ``WriteRefused`` and ``NotEnoughSpace`` (core/import_v1) without loading torch at import."""
    if name in ("Plan", "Job", "RunInfo"):
        return getattr(_r2(), name)
    if name in ("LayerImage", "ExportOptions", "LayerService"):
        from .render import layers
        return getattr(layers, name)
    if name in ("ImportReport", "WriteRefused", "NotEnoughSpace", "WriteGuard"):
        from .core import import_v1 as iv
        return getattr(iv, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
