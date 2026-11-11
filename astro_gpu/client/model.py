"""ProjectModel: the one adapter between the views and the engine API.

Each member is a thin call on ``astro_gpu.api``: the workflows (``WorkflowView``), their stages, keys and states, what a
change costs, the Replace / Try as variant choice, variants, undo and redo, the sessions, the selection thresholds and
the settings of a stage. Edits are saved by Save only (issue #1): until then the model is ``dirty``.

What the GUI thread may do, and what it may not:

* it edits the workflows, and reads keys, states, plans and costs, which the API answers from caches once ``warm`` has
  run: the model submits that to the ApiWorker when a project opens (``ready`` turns True, ``readyChanged`` fires);
* it never scans the disk, loads an array or computes an automatic value: ``describe`` (the settings of a stage), the
  counts of the sessions and the facts of the done results come from the worker. ``params(p, stage)`` answers from a
  cache and asks the worker for what it lacks; ``paramsChanged`` says when it arrives.

Vocabulary. The views use the client's own keys (French words): a workflow's ``opts`` are ``debruitage`` / ``dn``,
``etoiles`` (``garder``: the stars kept, ``retirer``: taken off), ``fond`` and ``deconv``; its blocks are
``calibration``, ``alignement``, ``selection``, ``empilement``, ``astrometrie``, ``couleur``, ``deconv``,
``debruitage``, ``etoiles``, ``fond``, ``post`` (blocks.py).
Stage names, settings and the stored ``params`` are the engine's. The model translates at the API's door
(``to_engine`` / ``from_engine``).

A change is a dict: ``{"kind": "param", "stage", "name", "value", "suggested"}`` (value None: back to the suggestion),
``{"kind": "opt", "key", "value"}`` (a method: ``dn``, ``fond``) or ``{"kind": "params", "items": [...]}``
(the Selection thresholds: two settings at once).
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from .. import api
from ..core import workflow as wfl
from ..core.project import Target
from ..core.project2 import FILE as PROJECT_FILE
from ..core.project2 import (ACQUISITION, FILTERS, Workflow, active_version, read_state, versions_problem,
                            workflow_images, write_state)
from .blocks import COLUMNS, KEY_ENGINE_BLOCK, TOOLS, TRUNK, block_state, kbase, knum  # noqa: F401  (TOOLS: test_model_surface)
from .blocks import ENGINE_BLOCK as ENGINE_KEY
from .i18n import _, hm
from .services.worker import ApiWorker, norm_path

#: the stage of the Output block: a workflow has a layer in Post once it is computed
BY_OUTPUT = ("render_simple",)
#: the name a layer of the Stars block's images gets after its workflow's ("Full · Stars and background"); its words
#: are translated where shown (post3.tr_name)
POST_PART_NAMES = {"starless": "Starless image", "stars": "Stars alone", "combined": "Combine layers"}

# ---------------------------------------------------------------------------------------------- vocabulary
#: the classic starting points, whose durations are asked of the worker beforehand (the gallery quotes them)
TEMPLATES = {
    "Vierge": dict(debruitage=False, etoiles="garder", fond="simple"),
    "Classique + débruitage": dict(debruitage=True, etoiles="garder", fond="simple"),
    "Complet": dict(debruitage=True, etoiles="retirer", fond="simple"),
}
#: the Denoise methods: the views' keys and the engine's
DN_ENGINE = {"generique": "pretrained", "affine": "adapted"}
DN_VIEW = {v: k for k, v in DN_ENGINE.items()}
CALIBRATION_KINDS = ("darks", "flats", "bias", "flat_darks")    # the session's lists a frame can be left out of
#: seconds of the methods the engine does not have yet (the spec's estimates)
EXTRA_SECONDS: dict[str, float] = {}
NOT_IN_ENGINE = "Not in the engine yet."
#: the block of the views → the engine's block of ``api.methods``, and the option each method sets
METHOD_BLOCK = {"synth": "synth", "debruitage": "denoise", "fond": "gradient", "etoiles": "stars", "deconv": "deconvolution"}
#: a Combine channels's stars are chosen in its panel: a Stars block in its graph was never drawn (GUI exploration 4)
FUSION_STARS = "Combine channels: its stars in its panel"
METHOD_OPT = {"synth": "synth_m", "debruitage": "dn", "fond": "fond", "etoiles": "etoiles", "deconv": "deconv_m"}
#: the engine's method keys → the views'
METHOD_KEY = {"debruitage": {"pretrained": "generique", "adapted": "affine"},
              "fond": {"polynomial": "simple", "rbf": "rbf", "automatic": "auto"},
              "etoiles": {"neural": "neural"},
              "deconv": {"rl": "psf", "neural": "neural"},
              "synth": {"measured": "measured", "instrument": "instrument"}}
#: the Deconvolution methods: the views' keys and the engine's
DECONV_ENGINE = {"psf": "rl", "neural": "neural"}
DECONV_VIEW = {v: k for k, v in DECONV_ENGINE.items()}
#: the Gradient methods: the views' keys and the engine's
GRADIENT_ENGINE = {"simple": "polynomial", "rbf": "rbf", "auto": "automatic"}
GRADIENT_VIEW = {v: k for k, v in GRADIENT_ENGINE.items()}
#: the toolbox: key of a tool → (engine option, value that adds it, value that removes it)
TOOL_OPTION = {"debruitage": ("denoise", "pretrained", None), "etoiles": ("stars", "neural", None),
               "deconv": ("deconvolution", "rl", None),
               "auto": ("gradient", "automatic", "polynomial"),
               "rbf": ("gradient", "rbf", "polynomial"),
               "simple": ("gradient", "polynomial", None)}
#: the stages whose settings the summaries of the blocks quote: asked of the worker for each workflow
PREFETCH = ("stack", "denoise")


def view_key(name: str) -> str:
    """The views' key of an engine block, or of a graph node's name (issue #47: "denoise~2" → "debruitage~2")."""
    b, sep, n = name.partition("~")
    return ENGINE_KEY.get(b, b) + (f"~{n}" if sep else "")


def engine_name(key: str) -> str:
    """The engine's name of a view's key ("debruitage~2" → "denoise~2", "alignement" → "registration")."""
    b, sep, n = key.partition("~")
    return KEY_ENGINE_BLOCK.get(b, b) + (f"~{n}" if sep else "")


def node_key(n: dict) -> str:
    """The views' key of a graph node: its name for an image block (repeated ones are numbered), its block else."""
    return view_key(n["name"] if n["block"] in wfl.IMAGE_OPS else n["block"])


def refused_text(e: Exception) -> str:
    """Why a block cannot be placed, removed or switched off, translated (its fields too)."""
    if isinstance(e, wfl.BlockRefused):
        return _(e.template).format(**{k: _(v) for k, v in e.fields.items()})
    return _(str(e))


class NotInEngine(ValueError):
    """What was asked is ◇: the engine does not have it yet."""


def to_engine(o: dict) -> dict:
    """The views' options of a workflow as the engine's (core/workflow). "cut": the engine's names of the blocks the
    user cut (issue #43), kept as they are."""
    stars = "neural" if o.get("etoiles") == "retirer" else None
    out = {"denoise": DN_ENGINE.get(o.get("dn"), "pretrained") if o.get("debruitage") else None, "stars": stars,
           "gradient": GRADIENT_ENGINE.get(o.get("fond"), "polynomial"),
           "deconvolution": DECONV_ENGINE.get(o.get("deconv_m"), "rl") if o.get("deconv") else None}
    if o.get("cut"):
        out["cut"] = list(o["cut"])
    return out


def from_engine(o: dict) -> dict:
    """The engine's options as the views'."""
    return {"debruitage": bool(o.get("denoise")), "dn": DN_VIEW.get(o.get("denoise"), "generique"),
            "etoiles": "retirer" if o.get("stars") else "garder", "fond": GRADIENT_VIEW.get(o.get("gradient"), "simple"),
            "deconv": bool(o.get("deconvolution")), "deconv_m": DECONV_VIEW.get(o.get("deconvolution"), "psf"),
            "cut": list(o.get("cut") or [])}


def is_fusion_view(p) -> bool:
    """A workflow that combines other workflows' outputs (issue #43)."""
    src = getattr(p, "source", None)
    return isinstance(src, dict) and "fusion" in src


def is_stretched_view(p) -> bool:
    """A workflow that starts from an image already stretched."""
    src = getattr(p, "source", None)
    return isinstance(src, dict) and bool(src.get("stretched"))


def engine_block(key: str) -> str:
    """The engine's block of a view's block key ("alignement" → "registration")."""
    return KEY_ENGINE_BLOCK.get(key, key)


def cuttable(key: str) -> bool:
    """Whether the user can cut this block (issue #43: Calibration, Registration, Astrometry, Gradient, Color)."""
    return engine_block(key) in wfl.CUTTABLE


def methods_of(block: str) -> list[dict]:
    """The methods of a block of the views, as the panel's cards show them: key (the views'), name, short, runs
    (gpu | cpu | both), what, new (◇), stage (the engine's)."""
    return [dict(key=METHOD_KEY[block][m.key], name=m.name, short=m.short, runs=m.runs_on, what=m.what, new=m.new,
                 stage=m.stage, engine_key=m.key) for m in api.methods(METHOD_BLOCK[block])]


def method_of(p: WorkflowView, block: str) -> str:
    if "~" in block:                      # a block placed twice ("fond~2"): its own node's method (round 3)
        base = block.partition("~")[0]
        m = getattr(p, "model", None)
        n = next((x for x in m.graph(p) if node_key(x) == block), None) if hasattr(m, "graph") else None
        if n is not None and n.get("method"):
            return METHOD_KEY[base].get(n["method"], n["method"])
        block = base
    if block == "fond":
        return p.opts["fond"]
    if block == "etoiles":
        return "neural"
    if block == "synth":
        m = getattr(p, "model", None)
        n = next((x for x in m.graph(p) if x["block"] == "synth"), None) if hasattr(m, "graph") else None
        return (n or {}).get("method") or "measured"
    if block == "deconv":
        return p.opts.get("deconv_m", "psf")
    return p.opts.get("dn", "generique")


def method_value(block: str, key: str):
    """The option value that selects a method."""
    if block == "etoiles":
        return "retirer"
    return key


@dataclass
class WorkflowView:
    """A workflow as the views see it: ``pid`` is the workflow's id."""

    pid: int
    name: str
    opts: dict
    params: dict = field(default_factory=dict)           # stage -> {setting: value} the user stored
    end: str | None = None                               # the block the layer leaves from ("empilement"), or None
    variant_of: int | None = None
    computed: bool = False                               # every stage is done
    rendered: bool = False                               # has a result to show in Post (now or earlier)
    layer: str = ""                                      # not read by the views (the layers are the engine's)
    provisional: bool = False
    source: dict | None = None                           # issue #43: the image it starts from, or None (the sessions)
    sessions: list | None = None                         # issue #43: the sessions it stacks (a filter's); None: all
    model: Any = field(default=None, repr=False, compare=False)

    @property
    def id(self) -> int:
        return self.pid

    def __deepcopy__(self, memo):
        """A copy shares the model (a QObject is never copied)."""
        return WorkflowView(self.pid, self.name, copy.deepcopy(self.opts, memo), copy.deepcopy(self.params, memo),
                            self.end, self.variant_of, self.computed, self.rendered, self.layer, self.provisional,
                            self.model)


@dataclass
class Pause:
    """A review a run stops at: the block of the views, the workflows that pause after it, and whether it is already
    computed (then there is nothing to run before the review). ``later``: the blocks that will pause the run after
    this one, in the graph's order."""

    block: str
    pids: list[int]
    done: bool = False
    later: list[tuple[int, str]] = field(default_factory=list)      # (workflow id, block)
    #: Run all: every workflow's own first review (workflow id, block); the first is ``block``. Empty: (pids, block)
    stops: list[tuple[int, str]] = field(default_factory=list)
    #: the workflows Run all holds back until the review (they would compute past a paused block)
    waiting: list[int] = field(default_factory=list)

    def all_stops(self) -> list[tuple[int, str]]:
        return list(self.stops) or [(pid, self.block) for pid in self.pids]


#: the reviews kept in state.json (the newest): enough for any project, small for the file
REVIEWED_KEPT = 400


@dataclass
class TargetState:
    """What the Project block says of the target. ``dot``: "known", "unknown" or "empty"; ``name``: what it shows;
    ``replace``: the object proposed in place of a target the solved field does not show."""

    dot: str
    name: str
    note: str
    replace: str = ""


@dataclass
class ProjectState:
    """How complete the Project block is (issue #89): ``level`` "none" (no target: red), "partial" (a target, shooting
    details missing: orange) or "complete" (green); ``missing``: the words of what is missing, in the form's order."""

    level: str
    missing: list[str]

    @property
    def line(self) -> str:
        """What is missing, in one line: the block's tooltip and the top of its form say it."""
        if not self.missing:
            return _("Complete: target and shooting details given")
        return _("Missing: {what}").format(what=", ".join(self.missing))


@dataclass
class Facts:
    """What the done results say, read by the worker: the metrics of the stages the views quote."""

    inventory: dict = field(default_factory=dict)
    register: dict = field(default_factory=dict)
    stack: dict = field(default_factory=dict)
    sessions: dict = field(default_factory=dict)         # session id -> {kind: file count}
    files: dict = field(default_factory=dict)            # session id -> {kind: [file path]} (issue #58)
    reviewed: list = field(default_factory=list)         # the results the user marked as checked (state.json)
    #: what makes a session's data impossible to compute, said before the run (``session_checks``): session id ->
    #: [(English template, its fields)]
    checks: dict = field(default_factory=dict)
    #: session id -> io.frames.colour_session of its lights (issue #57), None for raw or FITS mosaics
    colour: dict = field(default_factory=dict)
    #: session id -> the calibration kinds it gives as masters only (io/frames.master_kinds: their headers, read here)
    masters: dict = field(default_factory=dict)
    #: the named objects of the first field solved, biggest first (core/skyobjects.SkyObject)
    in_field: list = field(default_factory=list)


# ---------------------------------------------------------------------------------------------- worker jobs
def _warm_job(p) -> tuple[dict, Facts]:
    """Fill every disk cache of the API, the machine and the timing history (worker thread)."""
    from ..core import estimate as est
    api.warm(p)
    api.machine()
    return est.load_history(), _facts_job(p)


def _facts_job(p) -> Facts:
    from ..core.project import list_frames
    f = Facts()
    for w in p.workflows:
        for stage in ("inventory", "register", "stack"):
            if getattr(f, stage):
                continue
            try:
                out = api.stage_outputs(p, w.id, stage) or api.stage_outputs(p, w.id, stage, "previous")
            except (OSError, ValueError, KeyError):
                out = None
            if out is not None:
                setattr(f, stage, dict(out.metrics))
    for s in p.sessions:
        counts, files = {}, {}
        for kind in ("lights", "darks", "flats", "bias", "flat_darks"):
            try:
                got = list_frames([p.abspath(x) for x in getattr(s, kind)]) if getattr(s, kind) else []
            except (OSError, ValueError):
                got = []
            counts[kind] = len(got)
            files[kind] = [Path(x).as_posix() for x in got]
        f.sessions[s.id] = counts
        f.files[s.id] = files
        f.checks[s.id] = session_checks(p, s, files)
        f.colour[s.id] = _colour_of(files)
        f.masters[s.id] = _masters_of(files)
    if p.root is not None:
        f.reviewed = [str(x) for x in read_state(p.root).get("reviewed") or []]
    for w in p.workflows:
        try:
            f.in_field = list(api.sky_objects(p, w.id))
            break
        except Exception:  # noqa: BLE001 - not solved, no astrometry in it, no catalogue: nothing to say from it
            continue
    return f


def _colour_of(files: dict) -> dict | None:
    """io.frames.colour_session of a session's lights; None when it cannot be told (a damaged header: said elsewhere)."""
    from ..io.frames import colour_session
    try:
        return colour_session(files.get("lights") or [], _is_colour)
    except (OSError, ValueError):
        return None


def _masters_of(files: dict) -> set:
    """io.frames.master_kinds of a session's files (worker thread): a master beside raw lights is no file mistake."""
    from ..io.frames import master_kinds
    return master_kinds(files)


#: whether a frame is already in colour, by (path, modification time, size): its header is read once per client
_COLOUR_READS: dict = {}


def _is_colour(path) -> bool:
    from ..io.frames import is_colour_frame
    try:
        st = Path(path).stat()
    except OSError:
        return False
    k = (str(path), st.st_mtime_ns, st.st_size)
    if k not in _COLOUR_READS:
        _COLOUR_READS[k] = is_colour_frame(path)
    return _COLOUR_READS[k]


#: what the cheap reads of a frame said, by (path, modification time, size): a frame is read once per client
_FRAME_READS: dict = {}
#: the frame lists a session check names ("the folder of the darks")
FRAME_KINDS = {"lights": "lights", "darks": "darks", "flats": "flats", "bias": "bias", "flat_darks": "flat darks"}


def _frame_read(path: str, size: bool, colour: bool = False) -> tuple[bool, tuple[int, int] | None]:
    """(whether the frame can be read, its size (width, height) or None). Without ``size`` only a FITS file's first
    card is read (it must be SIMPLE: a text file renamed .fits is not a FITS file; a camera raw file is not opened);
    with it, the size from the header (NAXIS1/NAXIS2, or the raw header). Never the pixels of a FITS file; remembered
    per file. ``colour``: the session's lights are already in colour (issue #57), so its JPEG, PNG and TIFF files are
    frames: their header is read (elsewhere they are not read at all)."""
    from ..core.project import frame_kind
    try:
        st = Path(path).stat()
    except OSError:
        return False, None
    k = (path, st.st_mtime_ns, st.st_size, size, colour)
    if k in _FRAME_READS:
        return _FRAME_READS[k]
    out: tuple[bool, tuple[int, int] | None] = (True, None)
    kind = frame_kind(path)
    try:
        if kind == "fits":
            with open(path, "rb") as fh:
                head = fh.read(30)
            if not head.startswith(b"SIMPLE") and not head.startswith(b"\x1f\x8b"):    # gzipped: read by the engine
                out = (False, None)
            elif size and head.startswith(b"SIMPLE"):
                from ..io.frames import fits_header, useful_area
                hd = fits_header(path)
                area = useful_area(hd)                 # the size the engine reads the frame at
                w, h = area[2:] if area else (int(hd.get("NAXIS1", 0)), int(hd.get("NAXIS2", 0)))
                out = (True, (w, h) if w and h else None)
        elif kind == "raw" and size:
            try:
                from ..io.frames import read_info
                info = read_info(path)
                out = (True, (int(info.width), int(info.height)))
            except Exception:  # noqa: BLE001 - a raw file the decoder refuses: its size is not known, the engine says
                out = (True, None)
        elif kind == "colour" and colour:
            from ..io.frames import read_info
            info = read_info(path)
            out = (True, (int(info.width), int(info.height)) if size else None)
    except Exception:  # noqa: BLE001 - a FITS frame whose header cannot be read is said so
        out = (False, None)
    _FRAME_READS[k] = out
    return out


def session_checks(p, s, files: dict) -> list[tuple[str, dict]]:
    """What makes a session's data impossible to compute, found before the run (worker thread; PRINCIPLES: nothing
    runs without data): a folder of its frames that is not there (a disk unplugged), a FITS frame that is not a FITS
    file, calibration frames of another size than the lights. (English template, its fields); the kinds are
    translated where shown. Cheap reads only: the headers, the first frame of each kind for its size, each FITS file's
    first card."""
    out: list[tuple[str, dict]] = []
    for kind in FRAME_KINDS:
        for x in getattr(s, kind) or []:
            if not Path(p.abspath(x)).exists():
                out.append(("{session}: the folder of the {kind} is not found: {folder} (is its disk plugged in?)",
                            dict(kind=FRAME_KINDS[kind], folder=str(x))))
                break
    colour = _colour_of(files) is not None
    sizes: dict[str, tuple[int, int]] = {}
    for kind in FRAME_KINDS:
        bad: list[str] = []
        for path in files.get(kind) or []:
            ok, size = _frame_read(path, kind not in sizes, colour)   # the first readable frame says its kind's size
            if not ok:
                bad.append(path)
                if kind != "lights":
                    break
                continue                  # every light read: the Session says how many it uses (round 3: "5 / 5")
            if size is not None and kind not in sizes:
                sizes[kind] = size
        if bad:
            out.append(("{session}: {name} is not a frame that can be read (damaged, or not a FITS file): take it "
                        "out of the session", dict(name=Path(bad[0]).name, frames=kind, unread=len(bad))))
    if files.get("lights"):                  # issue #57: frames already in colour against mosaics, either way round
        for kind in ("darks", "flats", "bias", "flat_darks"):
            first = (files.get(kind) or [None])[0]
            if first is not None and _is_colour(first) != colour:
                out.append(("{session}: the {kind} ({name}) are not of the same kind as the lights (frames already in "
                            "colour against a raw mosaic): they cannot calibrate them; use {kind} of their kind",
                            dict(kind=FRAME_KINDS[kind], name=Path(first).name)))
    light = sizes.get("lights")
    for kind, size in sizes.items():
        if light is not None and kind != "lights" and size != light:
            out.append(("{session}: the {kind} are {w}×{h} pixels, the lights {lw}×{lh}: they cannot calibrate them",
                        dict(kind=FRAME_KINDS[kind], w=size[0], h=size[1], lw=light[0], lh=light[1])))
    return out


def _describe_job(p, wid: int, stage: str):
    key = api.stage_key(p, wid, stage)
    return wid, stage, key, api.describe(p, wid, stage)


def _selection_job(p, wid: int, tmin: float, ratio: float):
    t = api.selection_preview(p, wid, tmin, ratio)
    return wid, round(tmin, 6), round(ratio, 6), t.kept, t.total, [r.status != "" for r in t.rows]


def _sig(rows) -> tuple:
    return tuple((r.name, repr(r.value), repr(r.suggested), r.origin, r.reason, repr(r.reason_args), r.drift,
                  r.locked) for r in rows or [])


class SessionSet:
    """Which sessions a project has, for Undo (issue #47), with their names, filters and lists of files (control round,
    decision 7: a file taken out of a session, or added, is one gesture, one undo step; before, Ctrl+Z undid an older
    step and the file stayed out); it keeps a copy of each session, and of its frames excluded by hand, to bring back
    one that was removed. The frames excluded by hand are not undone here (Frames has its own way back)."""

    KINDS = ("lights", "darks", "flats", "bias", "flat_darks")

    def __init__(self, p):
        self.ids = [s.id for s in p.sessions]
        # renamed, its filter changed (2.3), a file taken out or added: undone
        self.labels = [(s.id, s.name, s.filter, *(tuple(getattr(s, k)) for k in self.KINDS)) for s in p.sessions]
        self.copies = {s.id: copy.deepcopy(s) for s in p.sessions}
        self.excluded = [list(x) for x in p.excluded]

    def __eq__(self, other):
        return isinstance(other, SessionSet) and self.labels == other.labels

    __hash__ = None

    def put_back(self, p) -> bool:
        """These sessions in ``p``: a removed one comes back as it was, an added one goes; the others get their name,
        filter and lists of files back. Whether any came or went, or their files changed (read again)."""
        files = False
        for s in p.sessions:
            if s.id in self.copies:
                c = self.copies[s.id]
                s.name, s.filter = c.name, c.filter
                for k in self.KINDS:
                    if list(getattr(s, k)) != list(getattr(c, k)):
                        setattr(s, k, list(getattr(c, k)))
                        files = True
        if [s.id for s in p.sessions] == self.ids:
            return files
        now = {s.id: s for s in p.sessions}
        p.sessions = [now.get(i) or copy.deepcopy(self.copies[i]) for i in self.ids]
        back = set(self.ids) - set(now)
        p.excluded = [x for x in p.excluded if x[0] in self.ids] + [x for x in self.excluded if x[0] in back]
        return True


class ProjectModel(QObject):
    pathsChanged = Signal()                # the workflows, their options or settings, or what is computed, changed
    currentChanged = Signal()              # the current workflow or the selected block
    costChanged = Signal()                 # what is to compute changed, without a new workflow or option
    sourcesChanged = Signal()              # the sessions or the frames changed
    testChanged = Signal()                 # the test area (Preview on a region, ◇)
    readyChanged = Signal()                # the disk caches are filled: keys, states and plans may be asked
    paramsChanged = Signal(int, str)       # the settings of this stage of this workflow arrived (workflow id, stage)
    factsChanged = Signal()                # the facts of the done results (info) changed
    selectionChanged = Signal()            # the count of kept frames changed
    postChanged = Signal(object)           # the Post composition changed (the object that changed it, or None)
    dirtyChanged = Signal(bool)            # there are edits not saved yet (True), or they were saved (False)
    editMade = Signal()                    # any edit of the definition: the recovery copy is written some time after

    def __init__(self, project, worker: ApiWorker | None = None, *, readonly: bool = False,
                 history: list | None = None, gpu: bool = True, device: str | None = None):
        super().__init__()
        self.p = project                                   # the definition (api.Project2): the source of truth
        self.root = Path(project.root) if project.root is not None else None
        self.readonly = readonly                           # never write the project (tests, fixtures)
        self._disk = self._disk_sig()                      # the file as this window read it (``flush``)
        self._own_worker = worker is None
        self.worker = worker or ApiWorker(self)
        self.worker.done.connect(self._worker_done)
        self.worker.failed.connect(self._worker_failed)
        self.paths: list[WorkflowView] = []
        self.current = 0
        self.sel_step = "sources"
        self.sel_layer: int | None = None                  # the Post layer of the selected Stars and background block
        self.scope: list[int] | None = None
        self.pending_change: dict | None = None
        self.gpu = gpu
        self.device = device if device else (None if gpu else "cpu")
        self.history = history if history is not None else []
        self.ready = False
        self.error = ""                                    # why the project could not be read
        self.test_on = False
        self.test_zone = (0.25, 0.25, 0.6, 0.6)
        self._facts = Facts()
        self._history_est: dict | None = None
        self._states: dict = {}
        self._chains: dict[int, list[str]] = {}
        self._blocks: dict[int, list] = {}
        self._cache: dict = {}                             # costs and plans, forgotten at each sync
        self._est: dict = {}                               # durations by stage key (kept across edits; the worker
        #                                                    fills it ahead: estimates read manifests)
        self._est_wanted = False                           # a view asked for a duration the worker had not read yet
        self._pc: dict[tuple[int, str], dict] = {}         # the settings of a stage of a workflow (describe)
        self._reasons: dict[tuple[str, str], tuple] = {}   # the last reason an automatic value had
        self._sel: dict | None = None                      # the last selection preview
        self._sel_want = None                              # what the last preview was asked for
        self._epoch = 0                                    # bumped when results change on disk
        self._reviewed: list[str] = []                     # "<block>:<fingerprint>" of the results marked as checked
        self._undo: list = []
        self._redo: list = []
        self._undo_what: list = []                         # what each undo step did, in words (None: not said)
        self._redo_what: list = []
        self._naming = False                               # the last step still takes its words (the same gesture)
        self.restored: str = ""                            # what the last Undo / Redo put back, in words
        self._depth = 0
        self._dirty = False
        self._post_removed_from_edited()
        self._rebuild_views()
        # the workflow and block shown when the project was last closed or saved (kept in the settings, so that a
        # project opened and closed without an edit is never written; GUI exploration 2), else as saved in it
        mem = self._remembered_selection()
        if self.paths:
            cur = mem.get("current", project.current)
            self.current = max(0, next((i for i, v in enumerate(self.paths) if v.pid == cur), 0))
        # GUI exploration 6.6: the block selected when the project was saved, if it is still there; else the Session
        # (a Stacking was selected, often not in the graph). The selection is kept with the next save, as ``current``
        sids = [s.id for s in project.sessions]
        sep = project.separate_sessions()
        w = project.workflow(self.paths[self.current].pid) if self.paths else None
        have = {"sources", "final"} | {f"{k}@{i}" for i in sep for k in ("session", "calibration")}
        if w is not None:
            have |= {"post"} | ({node_key(n) for n in w.blocks} if w.blocks is not None else
                                {"calibration", "alignement", "selection", "empilement"})
            if w.source is not None:                    # an Image or a Combine channels selected is taken back too
                have |= {"fusion"} if "fusion" in w.source else {"image"}
            # a Combine layers block (a Post layer, not a node; GUI exploration 4: it reopened on the Session): its stored
            # layer, the one remembered first; none stored yet, the workflow's default one
            if any(n["block"] == "stars" for n in self.graph(self.paths[self.current])):
                mine = [x for x in self.post_layers() if x.get("kind") == "workflow" and x.get("pid") == w.id]
                lids = [x.get("id") for x in mine if x.get("part") == "combined"]
                if lids or not mine:
                    have.add("composition")
                    self.sel_layer = mem.get("layer") if mem.get("layer") in lids else (lids[0] if lids else None)
        saved = mem.get("block", getattr(project, "current_block", ""))
        first = "image" if w is not None and w.source is not None and "fusion" not in w.source else "sources"
        self.sel_step = saved if saved in have else (f"session@{sep[0]}" if sep and sids else first)
        if self.sel_step != "composition":
            self.sel_layer = None
        # a method, not a lambda: a lambda holding the model, connected to its own signal, kept every closed project
        # in memory (GUI exploration 4)
        self.currentChanged.connect(self._keep_block)
        self._start_warm()

    def _keep_block(self):
        self.p.current_block = self.sel_step or ""

    # ------------------------------------------------------------ the definition, as the views see it
    @property
    def name(self) -> str:
        return self.p.name

    @property
    def target(self) -> str:
        return self.p.target.name or ""

    @property
    def path(self) -> WorkflowView | None:
        return self.paths[self.current] if 0 <= self.current < len(self.paths) else None

    @property
    def stack_mode(self) -> str:
        return self.p.stack_mode

    @property
    def sources_loaded(self) -> bool:
        return any(s.lights for s in self.p.sessions)

    def has_data(self, p: WorkflowView | None = None) -> bool:
        """Whether there is something to compute (issue #48): lights in a session, or the file of an Image block;
        for ``p``: its own source (a fusion: its channels')."""
        if p is None:
            return self.sources_loaded or any(bool((q.source or {}).get("path")) for q in self.paths)
        src = p.source
        if src is None:
            return self.sources_loaded
        if "fusion" in src:                  # its channels are the project's sessions: no lights, nothing (round 3: Shift+F5
            return self.has_data()           # on a Combine channels opened the run window where F5 said why in the status line)
        return bool(src.get("path"))

    @property
    def sessions(self) -> list[dict]:
        """The sessions as the views list them: id, name, night, and the count of each kind of frame."""
        out = []
        for s in self.p.sessions:
            c = self._facts.sessions.get(s.id, {})
            lights = c.get("lights", 0)
            ex = sum(1 for sid, _n in self.p.excluded if sid == s.id)
            out.append(dict(id=s.id, name=s.name, night=s.name, lights=lights, kept=max(lights - ex, 0),
                            flats=c.get("flats", 0), darks=c.get("darks", 0), bias=c.get("bias", 0),
                            dark_flats=c.get("flat_darks", 0), exposure=self._facts.inventory.get("exposure_s"),
                            iso=self._facts.inventory.get("iso_gain"), temp=None, filter=s.filter or "OSC", rotation=0.0,
                            fwhm=self._facts.register.get("fwhm_px"), flat_note=""))
        return out

    @property
    def frames_excluded(self) -> set[tuple[int, str]]:
        """The light frames excluded by hand: (session id, file name) pairs."""
        return {(int(a), str(b)) for a, b in self.p.excluded}

    def n_lights(self) -> int:
        n = sum(s["kept"] for s in self.sessions)
        return n or int(self._facts.inventory.get("n_lights") or 0)

    def index_of(self, pid: int) -> int:
        return next(i for i, q in enumerate(self.paths) if q.pid == pid)

    def view(self, pid: int) -> WorkflowView | None:
        return next((q for q in self.paths if q.pid == pid), None)

    def _rebuild_views(self):
        """The WorkflowViews from the definition; a view keeps its identity as long as its workflow exists."""
        by = {v.pid: v for v in self.paths}
        out = []
        for w in self.p.workflows:
            v = by.get(w.id) or WorkflowView(w.id, w.name, {}, model=self)
            v.name, v.opts = w.name, from_engine(w.options)
            v.params, v.variant_of = copy.deepcopy(w.params), w.variant_of
            v.end = "empilement" if w.end == "stacking" else None
            v.source = copy.deepcopy(w.source)
            v.sessions = list(w.sessions) if w.sessions is not None else None
            out.append(v)
        self.paths[:] = out
        self.current = min(max(self.current, 0), len(self.paths) - 1) if self.paths else -1

    def select(self, index: int | None = None, step: str | None = None):
        if index is not None:
            self.current = index
        if step is not None:
            self.sel_step = step
        v = self.path
        if v is not None and self.p.current != v.pid:
            self.p.current = v.pid                          # kept with the next save: not an edit
        self.currentChanged.emit()

    # ------------------------------------------------------------ the worker: warm, facts, settings
    def _start_warm(self):
        self.ready = False
        self.worker.submit(f"model.warm:{id(self)}", _warm_job, self.p)

    def _worker_done(self, channel: str, _n: int, result):
        me = f":{id(self)}"
        if channel == "model.warm" + me:
            self._history_est, self._facts = result
            self._take_reviewed(self._facts.reviewed)
            self.ready = True
            self.error = ""
            self._sync(prefetch=False)
            # the durations the views quote are read by the worker before the views are told (no manifest read in
            # the GUI thread when they draw)
            self.worker.submit(f"model.estimates:{id(self)}", self._prefetch_estimates, True)
        elif channel == "model.estimates" + me:
            if result:                                       # the first one, after warm: now the views may ask
                self.readyChanged.emit()
                self.sourcesChanged.emit()
                self.pathsChanged.emit()
        elif channel == "model.wanted" + me:                 # durations a view was waiting for have arrived
            self._est_wanted = False
            self.costChanged.emit()
        elif channel == "model.facts" + me:
            self._facts = result
            self._take_reviewed(result.reviewed)
            self._cache.clear()
            self.factsChanged.emit()
            self.costChanged.emit()
        elif channel.startswith("model.describe" + me + ":"):
            wid, stage, key, rows = result
            self._got_params(wid, stage, key, rows)
        elif channel == "model.selection" + me:
            new = dict(zip(("wid", "tmin", "ratio", "kept", "total", "excluded"), result))
            changed = self._sel is None or (new["kept"], new["total"]) != (self._sel["kept"], self._sel["total"])
            self._sel = new
            if changed:
                self.selectionChanged.emit()

    def _worker_failed(self, channel: str, _n: int, text: str):
        why = text.splitlines()[0] if text else "?"
        if channel == f"model.warm:{id(self)}":
            self.error = why
            self.readyChanged.emit()
        elif channel == f"model.estimates:{id(self)}":
            self.readyChanged.emit()                            # the durations will be read when asked
            self.sourcesChanged.emit()
            self.pathsChanged.emit()
        elif channel.startswith(f"model.describe:{id(self)}:"):
            _head, _id, pid, stage = channel.split(":", 3)
            v = self.view(int(pid))
            e = self._pc.get((int(pid), stage))
            if v is None or e is None:
                return
            e["asked"], e["tries"] = None, e.get("tries", 0) + 1
            if e["tries"] <= 2:                                  # an edit raced the worker: ask again, twice at most
                QTimer.singleShot(400, lambda: self.params(v, stage))
            else:
                e["error"] = why
                self.paramsChanged.emit(int(pid), stage)

    def reload_sources(self):
        """The sessions or the frames changed (files added, a new session): the definition is saved, the frames are
        scanned again, then the keys rebuilt. Until the worker is done the model answers from what it knew (no disk
        access in the GUI thread)."""
        self.edited()
        api.refresh(self.p, scan=True)
        self._cache.clear()
        self._est.clear()
        self._start_warm()

    def results_changed(self):
        """A stage ended or a run is over: the done results on disk changed. States, costs, the settings of the
        stages (their automatic values read the upstream results) and the facts are read again."""
        api.refresh(self.p)
        self._epoch += 1
        self._cache.clear()
        self._est.clear()
        if self.ready:
            self._sync()
            self.worker.submit(f"model.facts:{id(self)}", _facts_job, self.p)
        self.pathsChanged.emit()

    def mark_stages(self, items=None, emit: bool = True):
        """A run is over: the states are read again from disk (``results_changed``)."""
        self.results_changed()

    def set_machine(self, gpu: bool, device: str | None = None):
        """The machine the durations are for: with a GPU or without, and the device asked (cuda:N, cpu, None: auto)."""
        changed = (gpu, device if device else (None if gpu else "cpu")) != (self.gpu, self.device)
        self.gpu = gpu
        self.device = device if device else (None if gpu else "cpu")
        self._cache.clear()
        if self.ready and changed:
            self.worker.submit(f"model.estimates:{id(self)}", self._prefetch_estimates, False)
        self.costChanged.emit()
        self.pathsChanged.emit()

    def close(self, save: bool = True):
        """Save what is pending (unless ``save`` is False: the edits are dropped) and let go of the worker (the project
        is closed)."""
        if save:
            self.flush()
        else:
            self.remember_selection()                      # edits dropped: where it reopens is kept all the same
        self.worker.cancel(f"model.warm:{id(self)}")
        for sig, slot in ((self.worker.done, self._worker_done), (self.worker.failed, self._worker_failed)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        if self._own_worker:
            self.worker.stop()

    # ------------------------------------------------------------ saving
    @property
    def dirty(self) -> bool:
        """Edits not saved yet."""
        return self._dirty

    def _disk_sig(self):
        """The project's file as it is on the disk now (its time and size), None without one."""
        try:
            st = (Path(self.root) / PROJECT_FILE).stat()
        except (OSError, TypeError):
            return None
        return st.st_mtime_ns, st.st_size

    def changed_on_disk(self) -> bool:
        """The project's file was written by someone else (another window of the program) since this one read or
        wrote it: Save would replace that work (the night test pass: the last window to save won, silently)."""
        return self._disk is not None and self._disk_sig() != self._disk

    def flush(self, force: bool = False) -> bool:
        """Save: write the definition now (the edits are written by Save only). False, nothing written, when the file
        changed on the disk since this window read it (``changed_on_disk``), unless ``force``."""
        if self._dirty and not self.readonly and self.root is not None:
            if self.changed_on_disk() and not force:
                return False
            if self.path is not None:
                self.p.current = self.path.pid
            problem = versions_problem(self.p.post)
            if problem:                                   # the file is written all the same (issue #82)
                self.log("Post versions: {problem}", problem=problem)
            api.save(self.p)
            self._disk = self._disk_sig()
        self.remember_selection()                          # with or without an edit: where it reopens (2, 6.6)
        if self._dirty:
            self._dirty = False
            self.dirtyChanged.emit(False)
        return True

    #: the settings key of the block shown per project (its folder): {"current": workflow id, "block": block}
    SELECTION_KEY = "selection"

    def _remembered_selection(self) -> dict:
        if self.root is None:
            return {}
        from . import prefs
        got = (prefs.get(self.SELECTION_KEY) or {}).get(Path(self.root).as_posix().lower())
        return got if isinstance(got, dict) else {}

    def remember_selection(self):
        """The workflow and block shown, kept in the settings (never in the project: opening and closing it without
        an edit writes nothing into it): Save without an edit, and closing, keep where the project reopens."""
        if self.readonly or self.root is None:
            return
        from . import prefs
        want = {"current": self.path.pid if self.path is not None else None, "block": self.sel_step or ""}
        if self.sel_step == "composition" and self.sel_layer is not None:
            want["layer"] = self.sel_layer                  # which Combine layers block (GUI exploration 4)
        key = Path(self.root).as_posix().lower()
        cur = dict(prefs.get(self.SELECTION_KEY) or {})
        if cur.get(key) == want:
            return
        cur.pop(key, None)
        cur[key] = want                                    # the last one last; the oldest go beyond 50 projects
        prefs.set(self.SELECTION_KEY, dict(list(cur.items())[-50:]))

    def edited(self):
        """The definition changed (the workflows, the sessions, the frames excluded by hand, the Post composition):
        it waits for Save, the window says so."""
        if self.readonly:
            return
        self.editMade.emit()
        if self._dirty:
            return
        self._dirty = True
        self.dirtyChanged.emit(True)

    # ------------------------------------------------------------ the Post composition
    # Post's layers are the project's, in the active one of its versions (``project.post["versions"]``, issue #82). A
    # setting touches the active version; adding or taking out a workflow's image touches them all: ``scope`` says
    # which, with no default. Post edits them, and so
    # does the graph: the Stars block's images sent to Post, the Stars and background blocks of the Composition
    # section (layers of part "combined") and their settings in the graph's panel. Every change goes through
    # ``set_post_layers``, which says it with ``postChanged``: Post and the graph show the same composition.
    #: Post's layers as it shows them, when Post is open (``post3.PostView`` sets it): they hold the layers it gave
    #: the workflows, which are only saved with an edit
    post_shown = None

    def post_layers(self) -> list[dict]:
        """The composition: Post's as it shows it when it is open, else as saved (a copy: postengine.load_layers makes
        it safe to use)."""
        import copy as _copy
        if self.post_shown is not None:
            return self.post_shown()
        return _copy.deepcopy(list(self.post_version().get("layers") or []))

    def synth_wanted(self) -> dict[int, str]:
        """{workflow id: the method of its Synthetic stars block} for the workflows that have one."""
        out = {}
        for v in self.paths:
            n = next((x for x in self.graph(v) if x["block"] == "synth" and not x.get("off")), None)
            if n is not None:
                out[v.pid] = n.get("method") or "measured"
        return out

    def post_version(self) -> dict:
        """The active version of the composition."""
        return active_version(self.p.post)

    def post_versions(self) -> list[dict]:
        return list(self.p.post["versions"])

    def post_version_label(self) -> str:
        """The active version's name, for an export or the report; nothing when there is only one."""
        return self.post_version()["name"] if len(self.p.post["versions"]) > 1 else ""

    def post_scope(self, layers: list[dict]) -> str:
        """The scope of these layers replacing the active version's: all the versions when a workflow's image came or
        went."""
        return "active" if workflow_images({"layers": layers}) == workflow_images(self.post_version()) else "all"

    def set_post_version(self, vid: int):
        """Another version shown: not a step of the Undo (opening is never one)."""
        if vid != self.p.post["active"]:
            self.p.post = {**self.p.post, "active": vid}
            self.edited()
            self.postChanged.emit(None)

    def new_post_version(self) -> int:
        """A new version, a copy of the active one, made active. Its layers have ids of their own."""
        import copy as _copy

        from .postengine import reseed_ids, with_new_ids
        post = self.p.post
        with self._edit():
            reseed_ids([x for v in post["versions"] for x in v.get("layers") or []])
            have = {v["name"] for v in post["versions"]}
            n = len(post["versions"]) + 1
            while f"Version {n}" in have:
                n += 1
            vid = max(v["id"] for v in post["versions"]) + 1
            made = _copy.deepcopy(self.post_version())
            made.update(id=vid, name=f"Version {n}", layers=with_new_ids(made.get("layers") or []))
            self.p.post = {**post, "versions": [*post["versions"], made], "active": vid}
        self.postChanged.emit(None)
        return vid

    def rename_post_version(self, vid: int, name: str):
        name = name.strip()
        post = self.p.post
        if not name or any(v["name"] == name for v in post["versions"]):
            return
        with self._edit():
            self.p.post = {**post, "versions": [{**v, "name": name} if v["id"] == vid else v for v in post["versions"]]}
        self.postChanged.emit(None)

    def delete_post_version(self, vid: int):
        """A version taken out; the last one stays."""
        post = self.p.post
        rest = [v for v in post["versions"] if v["id"] != vid]
        if not rest:
            return
        with self._edit():
            self.p.post = {**post, "versions": rest, "active": post["active"] if post["active"] != vid else rest[-1]["id"]}
        self.postChanged.emit(None)

    def _post_sync(self):
        """Every version holds the workflows' images of the active one (a workflow touches them all, issue #82), each
        found by its workflow and its image; what the others lack comes with the active one's settings, and ids of its
        own."""
        from .postengine import reseed_ids, with_new_ids
        post = self.p.post
        act = active_version(post)
        want = workflow_images(act)
        reseed_ids([x for v in post["versions"] for x in v.get("layers") or []])
        for v in post["versions"]:
            if v is act:
                continue
            have = workflow_images(v)
            layers = list(v.get("layers") or [])
            for key, n in (have - want).items():
                mine = [x["id"] for x in layers if x.get("kind") == "workflow" and (x.get("pid"), x.get("part")) == key]
                gone = set(mine[-n:])
                layers = [x for x in layers if x["id"] not in gone and not (x.get("kind") == "filter"
                                                                            and x.get("target") in gone)]
            added = []
            for key, n in (want - have).items():
                mine = [x for x in act["layers"] if x.get("kind") == "workflow" and (x.get("pid"), x.get("part")) == key]
                added += mine[:n]
            v["layers"] = with_new_ids(added) + layers

    # A workflow's image is a layer of Post by default (the graph draws its wire, Post shows it). The images the user
    # took out (a wire removed, a layer deleted in Post) are listed in ``project.post["removed"]`` as (workflow id,
    # part) and do not come back by themselves (issues #10, #52); any other image, a new branch's or a new flow's,
    # still goes to Post.
    def _post_removed_from_edited(self):
        """A composition saved before the list (only "edited"): every workflow then without a layer was taken out."""
        post = getattr(self.p, "post", None) or {}
        if "removed" in post or not post.get("edited"):
            return
        have = {x.get("pid") for x in active_version(post).get("layers") or [] if x.get("kind") == "workflow"}
        self.p.post = {k: v for k, v in post.items() if k != "edited"}
        self.p.post["removed"] = [[w.id, part] for w in self.p.workflows if w.id not in have
                                  for part in (None, "combined")]

    def post_removed(self) -> set:
        """The (workflow id, part) taken out of Post by the user."""
        return {(int(a), b) for a, b in ((getattr(self.p, "post", None) or {}).get("removed") or [])}

    def post_default_part(self, p):
        """The part a workflow gives Post by default: with Stars, its Stars and background image."""
        return "combined" if p.opts.get("etoiles") == "retirer" else None

    def post_defaults(self, layers=None) -> list:
        """The workflows whose image Post shows by default with these layers (the saved ones by default): no layer of
        their own yet, an image to give, not a fusion's channel, not taken out."""
        if layers is None:
            layers = self.post_version().get("layers") or []
        have = {x.get("pid") for x in layers if x.get("kind") == "workflow"}
        out = self.post_removed()
        return [q for q in self.paths if q.pid not in have and (q.pid, self.post_default_part(q)) not in out
                and not self.feeds_fusion(q) and self.gives_post(q)]

    def set_post_layers(self, layers: list[dict], source=None, *, scope: str):
        """The composition replaced (saved layers, postengine.save_layers); ``source``: who changed it (Post itself
        does not reload what it wrote). An image shown before and not any more was taken out: it stays out. A change
        made from the graph is a step of its Undo (2.1); Post's own changes are in Post's Undo. ``scope``: "active", or
        "all" when a workflow's image came or went."""
        if scope not in ("active", "all"):
            raise ValueError(f"scope: active or all, not {scope!r}")
        if getattr(source, "is_post_page", False) or self._depth:
            self._set_post_layers(layers, source, scope)
            return
        with self._edit():
            self._set_post_layers(layers, source, scope)

    def _set_post_layers(self, layers: list[dict], source, scope: str):
        old = self.post_version().get("layers") or []
        shown = {(x.get("pid"), x.get("part")) for x in old if x.get("kind") == "workflow"}
        # Post holds layers only for the workflows already computed: the others' images, which it never had, are not
        # taken out by its saving its composition (GUI exploration 4: a filter added emptied Post of new branches)
        post = getattr(source, "is_post_page", False)
        shown |= {(q.pid, self.post_default_part(q)) for q in self.post_defaults(old)
                  if not post or getattr(q, "rendered", True)}
        now = {(x.get("pid"), x.get("part")) for x in layers if x.get("kind") == "workflow"}
        alive = {q.pid for q in self.paths}
        out = {x for x in (self.post_removed() | (shown - now)) - now if x[0] in alive}
        self.p.post = {**{k: v for k, v in self.p.post.items() if k != "edited"},
                       "removed": sorted([list(x) for x in out], key=str)}
        self.post_version()["layers"] = layers
        if scope == "all":
            self._post_sync()
        self.edited()
        self.postChanged.emit(source)

    def _post_edit(self, change, source=None, *, scope: str):
        from .postengine import load_layers, normalize, save_layers
        layers = load_layers(self.post_layers(), {q.pid for q in self.paths})
        out = change(layers)
        self.set_post_layers(save_layers(normalize(layers)), source, scope=scope)
        return out

    def add_post_layer(self, pid: int, part: str | None, source=None, keep_default: bool = True) -> int:
        """A layer of a workflow's image at the top of Post, shown: its Output block's image (``part`` None) or one of
        its Stars block's (the starless image, the stars alone, or a new Stars and background block). Returns its
        id. ``keep_default`` False: it is the layer Post shows by default for the workflow, made real (its first
        setting)."""
        from .postengine import new_workflow_layer
        p = self.view(pid)
        name = p.name + ("" if part is None else " · " + POST_PART_NAMES[part])
        lay = new_workflow_layer(pid, name, True, part)

        def change(layers):
            self._filled(layers, None if keep_default else pid)
            if keep_default and part is not None and any(q.pid == pid for q in self.post_defaults(layers)):
                # the image Post shows by default for this workflow stays: a Combine layers added is a second one, the
                # first double-click of its card included (2nd round)
                d = self.post_default_part(p)
                layers.insert(0, new_workflow_layer(pid, p.name + ("" if d is None else " · " + POST_PART_NAMES[d]),
                                                    True, d))
            same = [x for x in layers if x.get("kind") == "workflow" and x.get("pid") == pid and x.get("part") == part]
            if part == "combined" and same:               # several Combine layers of one workflow: numbered (2nd round)
                lay["name"] = f"{name} {len(same) + 1}"
            layers.insert(0, lay)
            return lay["id"]
        return self._post_edit(change, source, scope="all")

    def default_post_layer(self, pid: int) -> int:
        """The Combine layers Post shows by default for a workflow, made a layer of its own to open it in Post: what Post
        shows does not change, so it is not an undo step (decision 7: opening is never one)."""
        n, redo = len(self._undo), list(self._redo)
        lid = self.add_post_layer(pid, "combined", None, keep_default=False)
        del self._undo[n:]
        self._redo[:] = redo
        return lid

    def _filled(self, layers: list[dict], skip: int | None = None):
        """A composition never filled (no workflow layer yet): the layers Post gives the workflows first (with the
        split block, its Combine layers block; else its Output image: post3.sync_paths, the graph draws them before), so
        that an image sent on top does not take their place (issue #29). ``skip``: a workflow whose Combine layers block is
        the one being added."""
        from .postengine import new_workflow_layer
        if any(x.get("kind") == "workflow" for x in layers):
            return
        made = []
        for q in self.post_defaults(layers):                   # #47, #49, #52
            stars = q.opts.get("etoiles") == "retirer"
            if q.pid == skip and stars:
                continue
            made.append(new_workflow_layer(q.pid, q.name + (" · " + POST_PART_NAMES["combined"] if stars else ""),
                                           bool(q.end) or self.is_sent_branch(q),     # a block's image sent: shown,
                                           "combined" if stars else None))           # as the Split's (2nd round)
        if made:
            base = next((x for x in made if x["name"].startswith("Full")), made[-1])
            base["visible"] = True
            made.remove(base)
            layers[:0] = made + [base]

    def remove_post_part(self, pid: int, part: str | None, source=None) -> int:
        """A workflow's image taken out of Post: every layer showing it as it is (its own filters go with them);
        returns how many (issue #29: a link to Post the graph could not take away)."""
        def change(layers):
            self._filled(layers)                          # the layers Post gives by default, made real first (#52)
            gone = {x["id"] for x in layers if x.get("kind") == "workflow" and x.get("pid") == pid
                    and x.get("part") == part}
            layers[:] = [x for x in layers if x["id"] not in gone and not (x["kind"] == "filter"
                                                                           and x.get("target") in gone)]
            return len(gone)
        return self._post_edit(change, source, scope="all")

    def remove_post_layer(self, lid: int, source=None):
        """A layer taken out of Post (its own filters go with it)."""
        def change(layers):
            layers[:] = [x for x in layers if x["id"] != lid and not (x["kind"] == "filter" and x.get("target") == lid)]
        self._post_edit(change, source, scope="all")

    def set_post_layer(self, lid: int, source=None, **values):
        """Settings of a Post layer changed (``sliders``, ``blend``, ``visible``...)."""
        def change(layers):
            for x in layers:
                if x["id"] == lid:
                    x.update(values)
        self._post_edit(change, source, scope="active")

    def post_layer(self, lid: int) -> dict | None:
        return next((x for x in self.post_layers() if x.get("id") == lid), None)

    # ------------------------------------------------------------ the sync: states, chains, blocks
    def _sync(self, prefetch: bool = True):
        """The views, states, chains and block keys read again (GUI thread: caches and manifests only)."""
        self._rebuild_views()
        self._cache.clear()
        if not self.ready:
            return
        try:
            self._states = api.states(self.p, update=not self.readonly)
        except (OSError, ValueError, KeyError):
            self._states = {}
        self._chains, self._blocks = {}, {}
        for w in self.p.workflows:
            try:
                self._chains[w.id] = api.chain(self.p, w.id)
                self._blocks[w.id] = api.blocks(self.p, w.id)
            except (ValueError, KeyError):
                self._chains[w.id], self._blocks[w.id] = [], []
        for v in self.paths:
            stages = self._chains.get(v.pid, [])
            v.computed = bool(stages) and all(self.state(v, s) == "done" for s in stages)
            # the Output's image; an already stretched Image stands for it (its chain has no render_simple, the Image
            # block writes what the Output would: wfl.PROVIDES; engine_shapes: its layer never reached Post)
            out = next((s for s in reversed(stages) if s in BY_OUTPUT or ((v.source or {}).get("stretched") and any(
                o in BY_OUTPUT for o in wfl.PROVIDES.get(wfl.base(s), ())))), None)
            st = self._states.get((v.pid, out)) if out else None
            v.rendered = bool(st) and (st.state == "done" or bool(st.previous_key))
        for v in self.paths:
            for stage in PREFETCH:
                if stage in self._chains.get(v.pid, []):
                    self.params(v, stage)
        if prefetch:
            self.worker.submit(f"model.estimates:{id(self)}", self._prefetch_estimates, False)

    def _prefetch_estimates(self, first: bool) -> bool:
        """Worker thread: the durations the views quote (each stage of each workflow, the method cards, the toolbar
        cards, the New workflow dialog), into the estimates cache. Returns ``first``."""
        from .blocks import TOOLBOX
        for v in list(self.paths):
            for stage in list(self._chains.get(v.pid, [])):
                self.seconds(stage, v)
            # Deconvolution and Synthetic stars too (the night test pass: their cards said "…" for good)
            for block in ("debruitage", "etoiles", "fond", "deconv", "synth"):
                try:
                    for m in methods_of(block):
                        for dev in ("gpu", "cpu"):                 # the card says both: GPU and CPU
                            self.method_seconds(v, block, m["key"], dev)
                except (ValueError, KeyError):
                    pass
            for k in TOOLBOX:
                if not self.has_tool(v.opts, k):
                    try:
                        self.tool_cost(v.opts, k)
                    except (wfl.OptionRefused, ValueError):
                        pass
            trunk = {k: x for k, x in v.params.items() if k in TRUNK}
            for opts in TEMPLATES.values():                 # the New workflow dialog, from this workflow
                self.opts_seconds(opts, trunk)
            self.opts_seconds(v.opts, v.params)             # a copy of it
        for opts in TEMPLATES.values():
            self.opts_seconds(opts)
        return first

    @contextmanager
    def _edit(self):
        """One change of the workflows, for Undo: the first level takes the snapshot, and after it saves, reads the
        keys and states again and says what changed."""
        top = self._depth == 0
        before = self._snapshot() if top else None
        extra = self._extra() if top else None
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if top:
                self._sync_graphs()
                self._prune_breakpoints()
                self._prune_alignment()
                self._follow_branch_names()
                self._prune_post()
                if self._snapshot() != before:
                    self._undo.append(before)
                    del self._undo[:-200]
                    self._redo.clear()
                    self._new_step()
                    self.edited()
                elif self._extra() != extra:
                    self.edited()                           # not for Undo: the frames, not the workflows
                self._sync()
                self.pathsChanged.emit()

    # ------------------------------------------------------------ the Project block
    def set_target_name(self, name: str):
        """The target typed: its coordinates when the catalogues know it, else none (asked online when the field is
        solved, else the stars alone solve it)."""
        from ..stages.astrometry import find_offline
        name = name.strip()
        if name == (self.p.target.name or ""):
            return
        found = find_offline(name) if name else None
        with self._edit():
            self.p.target.name = name
            self.p.target.ra, self.p.target.dec = found if found is not None else (None, None)

    def set_target_values(self, **values):
        """The coordinates or the capture data typed (``ra``, ``dec``, ``focal_mm``, ``pixel_um``: None when empty)."""
        with self._edit():
            for k, v in values.items():
                setattr(self.p.target, k, v)

    def set_description(self, text: str):
        if text != (self.p.description or ""):
            with self._edit():
                self.p.description = text

    def set_project_name(self, name: str) -> bool:
        """The project's name typed in its form (issue #89): the top bar, the window's title and Home show it; its folder
        keeps its own. An empty name is refused (False), the old one kept."""
        name = name.strip()
        if not name:
            return False
        if name != self.p.name:
            with self._edit():
                self.p.name = name
        return True

    def set_shooting_detail(self, key: str, text: str):
        """A shooting detail (``ACQUISITION``) typed in the Report or in the Project block's form: one place in the
        project for both, a change Undo takes back as any other (issue #89); an empty one is forgotten."""
        text = text.strip()
        if text == self.p.acquisition.get(key, ""):
            return
        with self._edit():
            if text:
                self.p.acquisition[key] = text
            else:
                self.p.acquisition.pop(key, None)

    def project_state(self) -> ProjectState:
        """How complete the Project block is (issue #89): a target (typed, or what Astrometry found in the field), then
        every shooting detail the Report asks the user for (the files say the rest)."""
        words = dict(instrument=_("instrument"), mount=_("mount"), filter=_("filter"), location=_("location"),
                     date=_("date"))
        has_target = bool(self.target_state().name)
        missing = ([] if has_target else [_("target")]) + [words[k] for k in ACQUISITION if not self.p.acquisition.get(k)]
        return ProjectState("none" if not has_target else "partial" if missing else "complete", missing)

    def target_state(self) -> TargetState:
        """What the Project block says of the target: known, unknown or nothing; a target the solved field does not
        show is never replaced by itself, only proposed. The field's object is read from the results, not written into
        the project: what the user typed, which Astrometry starts from, stays as it is."""
        from ..stages.astrometry import _norm, find_offline
        t = self.p.target
        typed = (t.name or "").strip()
        field_ = self._facts.in_field
        main = field_[0] if field_ else None
        if not typed:
            if main is not None:
                return TargetState("known", main.name, _("Found by Astrometry in the field"))
            return TargetState("empty", "", _("Optional: Astrometry fills it in"))
        found = find_offline(typed)
        if found is None:
            if t.ra is not None and t.dec is not None:
                return TargetState("known", typed, _("Coordinates entered"))
            return TargetState("unknown", typed, _("Not in the catalogues here: it will be looked up online, else the "
                                                   "field is solved from its stars alone"))
        if field_ and not any(_norm(typed) in {_norm(x) for x in (o.name, o.common_name, *o.aliases) if x} for o in field_):
            return TargetState("unknown", typed, _("Not in the solved field, which shows {name}").format(name=main.name),
                               main.name)
        return TargetState("known", typed, _("Found: RA {ra}°, Dec {dec}°").format(ra=f"{found[0]:.3f}", dec=f"{found[1]:+.3f}"))

    def relocated(self, root):
        """The project was given its folder (it had none of its own yet)."""
        self.root = Path(root)
        self._disk = self._disk_sig()

    def join_session(self, p, sid: int) -> WorkflowView | None:
        """A new session joins the chosen flow of frames ``p`` (GUI exploration 2, decision 1): it is added to that
        flow's sessions (every workflow of the flow: its branches too), never put in their place; joining a filter's
        flow, a session without a filter takes that filter. A Combine channels or an Image chosen, a session of another filter,
        or nothing chosen among several flows: it starts a flow of its own (1.8: a Session placed is always in the
        graph). Returns the workflow that reads it (None when there is none)."""
        frames = [w for w in self.p.workflows if w.source is None]
        ses = self.p.session(sid)

        def ids(w):
            return None if w.sessions is None else tuple(sorted(w.sessions))
        flows = {ids(w) for w in frames if ids(w) is None or sid not in ids(w)}
        w = next((x for x in frames if p is not None and x.id == p.pid), None)
        if w is None and p is None and len(flows) == 1:   # nothing chosen, one flow: that one (decision 5)
            w = next(x for x in frames if ids(x) in flows)
        if w is not None and sid in (w.sessions or ()):
            return self._view_of(w.id)
        flt = {x.filter for x in self.p.sessions_of(w.id) if x.id != sid} if w is not None else set()
        like = None                                       # the flow an other filter's frames were refused from
        if w is not None and ses.filter and flt and flt != {ses.filter}:
            like = w
            w = next((x for x in frames if x.sessions and sid not in x.sessions    # OIII frames never join the
                      and {s.filter for s in self.p.sessions_of(x.id)} == {ses.filter}), None)   # H-alpha flow;
            if w is not None:                             # an OIII flow there: that one
                flt = {ses.filter}
        with self._edit():
            if w is None:                                 # a flow of its own: the flows reading every session keep
                mine = [x.id for x in self.p.sessions if x.id != sid]     # the ones they had
                for x in frames:
                    if x.sessions is None:
                        x.sessions = list(mine)
                # a filter's flow: its name, and the blocks of the flow it was kept out of (one chain per filter,
                # as the filters template makes them)
                blocks = json.loads(json.dumps(like.blocks)) if like is not None and like.blocks else []
                q = self.add_graph_workflow(ses.filter or _("Flow"), blocks, sessions=[sid])
                self.p.validate()
                return q
            if len(flt) == 1 and "" not in flt and not ses.filter:
                ses.filter = next(iter(flt))              # a filter's flow: its filter (decision 1)
                # a default name shows the filter when the filter's other nights' names do ("Ha 2", GUI exploration 4)
                mates = [x.name for x in self.p.sessions_of(w.id) if x.id != sid]
                default = ses.name.startswith(_("Session {n}").format(n="")) and ses.name.split()[-1].isdigit()
                if default and mates and all(ses.filter in x for x in mates):
                    k = len(mates) + 1
                    while f"{ses.filter} {k}" in {x.name for x in self.p.sessions}:
                        k += 1
                    ses.name = f"{ses.filter} {k}"
            if w.sessions is not None:
                flow = ids(w)
                for x in frames:
                    if ids(x) == flow:
                        x.sessions = list(x.sessions) + [sid]
            self.p.validate()
        return self._view_of(w.id)

    def _snapshot(self):
        """What Undo puts back: the workflows, and which sessions there are (a Session block is a block of the graph:
        removed, it comes back with what it had, issue #47), with their lists of files (``SessionSet``)."""
        post = json.dumps(getattr(self.p, "post", None) or {}, sort_keys=True)     # the composition too (2.1)
        # the Project block's form too (issue #89): its name and shooting details, as its target and comments
        return (api.workflows(self.p), self.p.current, self.p.stack_mode, SessionSet(self.p), post,
                asdict(self.p.target), self.p.description, self.p.name, dict(self.p.acquisition))

    def _extra(self):
        """What an edit may change besides the workflows: the sessions and the frames excluded by hand."""
        return (sorted(tuple(x) for x in self.p.excluded),
                [(s.id, s.name, s.filter, *(tuple(getattr(s, k)) for k in ("lights", "darks", "flats", "bias",
                                                                          "flat_darks")))
                 for s in self.p.sessions])

    # ------------------------------------------------------------ undo and redo
    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        what = self._move_what(self._undo, self._undo_what, self._redo_what)
        done = self._restore(self._undo, self._redo)
        self.restored = what if done else ""
        return done

    def redo(self) -> bool:
        what = self._move_what(self._redo, self._redo_what, self._undo_what)
        done = self._restore(self._redo, self._undo)
        self.restored = what if done else ""
        return done

    def _new_step(self):
        """A step was pushed for Undo: it takes the words of its gesture (``name_step``, or its first Log line) until
        the gesture ends (the event loop's next turn)."""
        self._undo_what.append(None)
        self._redo_what.clear()
        self._naming = True
        QTimer.singleShot(0, self._end_naming)

    def _end_naming(self):
        self._naming = False

    def name_step(self, text: str):
        """Say what the step just pushed did ("Gradient 1 désactivé · Workflow"), for the Undo / Redo message: the
        message the user saw for it, " · Ctrl+Z" taken off."""
        if self._naming and self._undo_what:
            self._undo_what[-1] = text.replace(" · Ctrl+Z", "").strip()
            self._naming = False

    @staticmethod
    def _move_what(stack: list, src: list, dst: list) -> str:
        """The words of the step Undo / Redo takes from ``stack`` (moved to the other list), "" when not said."""
        if not stack:
            return ""
        while len(src) < len(stack):                       # the two lists stay aligned on the steps
            src.insert(0, None)
        del src[:len(src) - len(stack)]
        what = src.pop()
        dst.append(what)
        if isinstance(what, tuple):                         # a Log line: its template and fields, translated now
            try:
                return _(what[0]).format(**what[1])
            except (KeyError, IndexError, ValueError):
                return what[0].format(**what[1])
        return what or ""

    def _shown_keys(self, p) -> set:
        """The keys of the blocks the graph draws for ``p`` (a session's blocks, a source, Combine layers included)."""
        from .blocks import Graph
        g = Graph(self)
        return {n.ref for n in g.nodes if p in n.paths or (n.sid is not None and n.key != "calibration")} | \
            {k for k in (self.sel_step,) if k and g.node_of(p, k) is not None}

    def _restore(self, src: list, dst: list) -> bool:
        """Put the workflows back as an earlier snapshot had them (computed results are never removed)."""
        if not src:
            return False
        dst.append(self._snapshot())
        had = {v.pid: self.block_keys(v) for v in self.paths}      # decision 5: what comes back is selected
        old_keys = self.block_keys(self.path) if self.path is not None else []
        ws, current, mode, sessions, post, target, description, name, acquisition = src.pop()
        self.p.target, self.p.description, self.p.name, self.p.acquisition = Target(**target), description, name, \
            dict(acquisition)
        frames_back = sessions.put_back(self.p)           # first: the workflows name them
        api.set_workflows(self.p, ws, current)
        self.p.stack_mode = mode
        post_back = json.loads(post) != (getattr(self.p, "post", None) or {})
        if post_back:
            self.p.post = json.loads(post)
        if frames_back:
            self.reload_sources()
        self.pending_change = None
        self.edited()
        self._sync()
        self._unpatch_rows()
        if self.paths:
            self.current = next((i for i, v in enumerate(self.paths) if v.pid == self.p.current), 0)
            # decision 5: a block brought back is selected; a block gone leaves the one before it selected
            back = [(i, k) for i, v in sorted(enumerate(self.paths), key=lambda iv: iv[0] != self.current)
                    for k in self.block_keys(v) if k != "post" and k not in had.get(v.pid, [])]
            keys = self.block_keys(self.path)
            if back:
                self.current, self.sel_step = back[0]
            elif self.sel_step and self.sel_step not in ("sources", "final") and self.sel_step not in keys and \
                    self.sel_step not in self._shown_keys(self.path):
                j = old_keys.index(self.sel_step) if self.sel_step in old_keys else len(old_keys)
                # a block the graph draws (control round: the engine's Selection, not drawn, was chosen: nothing lit)
                shown = self._shown_keys(self.path)
                drawn = [k for k in keys if k in shown]
                self.sel_step = next((k for k in reversed(old_keys[:j]) if k in drawn),
                                     drawn[-1] if drawn else "sources")
        self.pathsChanged.emit()
        self.currentChanged.emit()
        if post_back:
            self.postChanged.emit(None)
        return True

    # ------------------------------------------------------------ stages, keys, states
    def all_stages(self, p: WorkflowView | None = None) -> list[str]:
        p = p or self.path
        return list(self._chains.get(p.pid, [])) if p is not None else []

    def path_stages(self, p: WorkflowView | None = None) -> list[str]:
        return [s for s in self.all_stages(p) if s not in TRUNK]

    def block_keys(self, p: WorkflowView) -> list[str]:
        """The views' block keys of a workflow, in order (the frames and Post excluded; Output last). The blocks of
        each session (issue #43: "calibration@2") are not in it: the graph draws them before the chain."""
        return [view_key(b.block) for b in self._blocks.get(p.pid, []) if wfl.base(b.block) in ENGINE_KEY
                and "@" not in b.block]

    # ------------------------------------------------------------ issue #43: cut and greyed blocks, sessions
    def display_keys(self, p: WorkflowView) -> list[str]:
        """The blocks the graph draws for a workflow, in order (issue #47): the nodes of its graph, those switched off
        included (drawn so); a block the graph does not have is not drawn. Each session's Calibration is drawn
        before the chain when the sessions are calibrated each on its own."""
        if is_fusion_view(p):                              # a fusion: one block, in the Composition section
            return ["fusion"]
        try:
            w = self.p.workflow(p.pid)
            g = wfl.graph_of(w) if w.blocks is not None else None      # a workflow made before: as it was drawn
        except (KeyError, ValueError):
            g = None
        if g is not None and p.pid in self._chains:      # once the worker has read the project
            keys = ["image"] if p.source is not None else []
            keys += ["align_images"] if (p.source or {}).get("align") is not None else []
            sep = bool(self.separate_sessions()) and p.source is None
            keys += [node_key(n) for n in g if not (sep and n["block"] == "calibration")]
            if "etoiles" in keys and "post" in keys:     # drawn as always: the Stars/background block holds the Output (#17)
                keys.remove("post")
                keys.insert(keys.index("etoiles") + 1, "post")
            if is_stretched_view(p) and "post" not in keys:
                keys.append("post")
            return keys
        keys = self.block_keys(p)
        if is_stretched_view(p) and "post" not in keys:  # its Output: the image itself (no stage of its own)
            keys = keys + ["post"]
        if self.separate_sessions() and p.source is None:
            keys = [k for k in keys if k != "calibration"]      # each session's Calibration is drawn before the chain
        want = [k for k in ("image", "calibration", "alignement", "selection", "crop", "empilement", "astrometrie", "fond",
                            "couleur", "debruitage", "etoiles", "synth", "post")]
        have = set(keys)
        extra = {ENGINE_KEY.get(b, b) for b in self.cut_blocks(p)} | {ENGINE_KEY.get(b, b) for b in self.greyed(p)}
        if p.source is not None:
            extra -= {"calibration", "alignement", "selection", "empilement"}
            if is_stretched_view(p):
                extra -= {"astrometrie", "fond", "couleur", "debruitage"}
        elif self.separate_sessions():
            extra.discard("calibration")                   # each session's Calibration is drawn before the chain
        if p.end:
            extra = {k for k in extra if want.index(k) <= want.index(p.end)}
        out = list(keys)                                   # the engine's order, the extras at their places
        for k in sorted(extra - have, key=want.index):
            before = [i for i, x in enumerate(out) if x in want and want.index(x) < want.index(k)]
            out.insert(before[-1] + 1 if before else 0, k)
        return out

    def cut_blocks(self, p: WorkflowView) -> list[str]:
        """The engine's names of the blocks the user switched off in this workflow (issue #47)."""
        return [n["block"] for n in self.graph(p) if n["off"]]

    def greyed(self, p: WorkflowView) -> dict[str, str]:
        """Issue #47: none. A block that cannot run where it is cannot be placed there (the palette says why)."""
        return {}

    # ------------------------------------------------------------ issue #47: the graph of a workflow
    def graph(self, p: WorkflowView) -> list[dict]:
        """The nodes of a workflow's graph the user places (a copy): its Output is not one of them (issue #49: made
        automatically, put back where it belongs when the graph is stored)."""
        try:
            return [n for n in wfl.graph_of(self.p.workflow(p.pid)) if n["block"] != "output"]
        except (KeyError, ValueError):
            return []

    def source_kind(self, p: WorkflowView) -> str:
        return wfl.source_of(self.p.workflow(p.pid))

    def node_index(self, p: WorkflowView, key: str) -> int | None:
        """The index in the graph of the node a view's key names."""
        return next((i for i, n in enumerate(self.graph(p)) if node_key(n) == key), None)

    def palette_reason(self, p: WorkflowView | None, block: str) -> str:
        """Why the palette greys a block for this workflow (translated), "" when it can be placed. Without a
        workflow, a project with sessions starts one from the frames (the block is checked on an empty graph)."""
        if p is None:
            if not self.p.sessions:
                return _("place Session/Image first")
            e = wfl.palette_problem([], "frames", block)
            return refused_text(e) if e is not None else ""
        if block == "color" and self.is_mono(p):
            return _(wfl.MONO_NO_COLOUR)
        if block == "stars" and is_fusion_view(p):
            return _(FUSION_STARS)
        e = wfl.palette_problem(self.graph(p), self.source_kind(p), block)
        if e is not None and block == "calibration" and self._nights_as_one(p):
            return _("calibrate per night first")   # one Calibration of all the nights: none to set beside it
        return refused_text(e) if e is not None else ""

    def gives_post(self, p) -> bool:
        """Whether a workflow gives Post an image: one made before the free graph does; a graph does once it has an
        Output block (a stretched image is its own; issue #49)."""
        try:
            w = self.p.workflow(p.pid)
        except KeyError:
            return False
        if w.source is None and not any(s.lights for s in self.p.sessions_of(w.id)):
            return False                                  # no lights yet: no layer (GUI exploration, question 4)
        if isinstance(w.source, dict) and "path" in w.source and not w.source["path"]:
            return False                                  # an Image block without its file (3.5)
        if w.blocks is None or wfl.source_of(w) == "stretched":
            return True
        return any(n["block"] == "output" and not n["off"] for n in w.blocks)

    # ------------------------------------------------------------ the Align images block
    def image_flows(self) -> list[WorkflowView]:
        """The workflows that start from an image with its file: what Align images can take."""
        return [q for q in self.paths if isinstance(q.source, dict) and q.source.get("path")]

    def align_problem(self) -> str:
        """Why Align images cannot be used (translated), "" when it can."""
        return "" if len(self.image_flows()) >= 2 else _("needs 2 images")

    def align_group(self, p) -> list[WorkflowView]:
        """The images aligned together with ``p`` (its block's wires), the reference's first."""
        ref = (p.source or {}).get("align") if isinstance(p.source, dict) else None
        if ref is None:
            return []
        group = [q for q in self.paths if isinstance(q.source, dict) and q.source.get("align") == ref]
        return sorted(group, key=lambda q: q.pid != ref)

    def align_images(self) -> WorkflowView | None:
        """Every image in the Align images block, on the reference of one already there, else on the first; the first
        image, or None when there are not two."""
        flows = self.image_flows()
        if len(flows) < 2:
            return None
        ref = next((q.source["align"] for q in flows if q.source.get("align") is not None), flows[0].pid)
        with self._edit():
            for q in flows:
                w = self.p.workflow(q.pid)
                w.source = {**w.source, "align": ref}
        return self.view(flows[0].pid)

    def set_align_reference(self, p, pid: int) -> None:
        with self._edit():
            for q in self.align_group(p):
                w = self.p.workflow(q.pid)
                w.source = {**w.source, "align": pid}

    def remove_alignment(self, p) -> None:
        with self._edit():
            for q in self.align_group(p):
                w = self.p.workflow(q.pid)
                w.source = {k: v for k, v in w.source.items() if k != "align"}

    def _prune_alignment(self):
        """A block of one image, or on a workflow gone, has nothing to do: it goes."""
        ws = {w.id: w for w in self.p.workflows}
        marked = [w for w in ws.values() if isinstance(w.source, dict) and w.source.get("align") is not None]
        for w in marked:
            mates = [x for x in marked if x.source.get("align") == w.source["align"]]
            if w.source["align"] not in ws or len(mates) < 2:
                w.source = {k: v for k, v in w.source.items() if k != "align"}

    def is_plain_image(self, p) -> bool:
        """An image going to Post as it is: an Image block with a file, no block placed after it."""
        return isinstance(p.source, dict) and bool(p.source.get("path")) and all(
            n["block"] == "output" for n in self.graph(p))

    def awaits_post(self, p) -> bool:
        """Whether a workflow will give Post an image once its lights, or its image file, are there: its wire to Post
        is drawn dashed until then."""
        try:
            w = self.p.workflow(p.pid)
        except KeyError:
            return False
        missing = (w.source is None and not any(s.lights for s in self.p.sessions_of(w.id))) or \
            (isinstance(w.source, dict) and "path" in w.source and not w.source["path"])
        return missing and (w.blocks is None or any(n["block"] == "output" and not n["off"] for n in w.blocks))

    def is_graph(self, p) -> bool:
        """A workflow made as a graph (issue #47), not from options."""
        try:
            return self.p.workflow(p.pid).blocks is not None
        except KeyError:
            return False

    def feeds_fusion(self, p) -> bool:
        """Whether a workflow is a channel of a fusion (issue #47: its output is not a layer of Post by itself)."""
        return any(is_fusion_view(q) and p.pid in [i["pid"] for i in q.source["fusion"]["inputs"]] for q in self.paths)

    def fusions_of(self, p) -> list[WorkflowView]:
        """The fusions that read a workflow's output (it is one of their channels)."""
        return [q for q in self.paths if is_fusion_view(q) and p.pid in [i["pid"] for i in q.source["fusion"]["inputs"]]]

    def star_fusions(self, p) -> list[WorkflowView]:
        """The fusions whose stars come from a workflow (mode "workflow", issue #55)."""
        return [q for q in self.paths
                if is_fusion_view(q) and (q.source["fusion"].get("stars") or {}).get("pid") == p.pid]

    @staticmethod
    def _fusion_pids(f: dict) -> set[int]:
        """The workflows a fusion reads: its channels', and the one its stars come from."""
        st = f.get("stars") or {}
        return {i["pid"] for i in f.get("inputs") or []} | ({st["pid"]} if st.get("pid") else set())

    def channel_problem(self, p) -> str:
        """Why a fusion's channel or star source cannot go (its workflow, its only Session, its Image), translated; ""
        for a workflow no fusion reads (GUI exploration 2, decision 3: refused with the reason, never a broken
        project)."""
        fs = self.fusions_of(p)
        if not fs:
            st = self.star_fusions(p)
            return _("stars of “{fusion}”").format(fusion=st[0].name) if st else ""
        role = next(i["role"] for i in fs[0].source["fusion"]["inputs"] if i["pid"] == p.pid)
        return _("channel {role} of “{fusion}”").format(role=role, fusion=fs[0].name)

    def _fusion_need(self, p, blocks) -> str:
        """Why a channel's graph ``blocks`` (a block taken out, or switched off) would leave a fusion unable to run:
        the Astrometry that aligns the channels, the Stars/background of a fusion that takes their starless images
        (decision 3); "" when every fusion reading it still can."""
        on = {n["block"] for n in blocks if not n.get("off")}
        for q in self.fusions_of(p) + self.star_fusions(p):
            f = q.source["fusion"]
            several = len(self._fusion_pids(f)) > 1
            if several and "astrometry" not in on and any(
                    n["block"] == "astrometry" and not n.get("off") for n in self.graph(p)):
                return _("needed by “{fusion}”").format(fusion=q.name)
            if f.get("stars") and "stars" not in on and any(
                    n["block"] == "stars" and not n.get("off") for n in self.graph(p)):
                return _("needed by “{fusion}”").format(fusion=q.name)
        return ""

    def fusion_problem(self, p) -> str:
        """Why a fusion cannot run, said before Run (GUI exploration 1.1, 7.6, question 8: the block is circled in red
        and the run refused); "" when it can, or for a workflow that is not a fusion. A channel's own block names it."""
        if not is_fusion_view(p):
            return ""
        f = p.source["fusion"]
        ins = f.get("inputs") or []
        if not ins:
            return _("channels to choose")
        views = {q.pid: q for q in self.paths}
        several = len(self._fusion_pids(f)) > 1              # a star source elsewhere is aligned too (issue #55)
        st = f.get("stars") or {}
        found: list[tuple[str, tuple, list[str]]] = []      # (template, its other fields, the channels): every channel
        #                                                     said at once (round 2: "Ha" only, then "OIII" once fixed)
        for i, role in [(i, i.get("role") or "") for i in ins] + ([(st, _("Stars"))] if st.get("mode") == "workflow"
                                                                  else []):
            q = views.get(i["pid"])
            why = self._channel_problem(q, i, role, f, several)
            if why is None:
                continue
            same = next((x for x in found if x[0] == why[0] and x[1] == why[1]), None)
            if same is None:
                found.append((why[0], why[1], [role]))
            elif role not in same[2]:
                same[2].append(role)
        out = []
        for t, more, roles in found:
            who = ", ".join(roles)
            out.append(_(t).format(role=who, **dict(more)) if t else f"{who}: {dict(more)['text']}")
        return " · ".join(out)

    def _channel_problem(self, q, i, role, f, several) -> tuple[str, tuple] | None:
        """What a fusion's channel lacks: (English template with {role}, its other fields as pairs); the template
        '' for a text made already (``text``). None when it has what the fusion needs."""
        if q is None:
            return "{role}: no workflow", ()
        if q.source is None and not any(s.lights for s in self.p.sessions_of(q.pid)):
            return "{role}: no lights", ()
        flts = {s.filter for s in self.p.sessions_of(q.pid)} if q.source is None else set()
        if i.get("channel", "L") == "L" and role in FILTERS and flts and flts != {role}:
            # decision 3: a channel's session set to another filter: refused with the reason
            return "{role}: session filter {filter}", (("filter", ", ".join(sorted(x or _("none") for x in flts - {role}))),)
        if q.source is not None and "fusion" not in q.source and not q.source.get("path"):
            return "{role}: no image file", ()
        if self.image_problem(q):
            return "", (("text", self.image_problem(q)),)
        if self.source_kind(q) == "stretched":             # marked so after it was chosen
            return "{role}: stretched image", ()
        blocks = [n["block"] for n in self.graph(q) if not n.get("off")]
        if several and self.is_graph(q) and "astrometry" not in blocks:
            return "{role}: needs Astrometry", ()
        if f.get("stars") and self.is_graph(q) and "stars" not in blocks:
            return "{role}: needs Stars/background", ()
        return None

    def run_problems(self, pids=None) -> list[tuple[str, str]]:
        """(workflow name, reason) of what cannot run: the fusions that cannot (1.1) and the workflows whose data
        cannot be computed (``data_problems``). Run is refused while there is one. ``pids``: only these workflows
        (a plan's: Run to here on a channel's Stacking is not refused because of the fusion after it)."""
        views = [q for q in self.paths if pids is None or q.pid in pids]
        return [(q.name, why) for q in views if (why := self.fusion_problem(q) or self.image_problem(q))] + \
            self.data_problems(views)

    def image_problem(self, p) -> str:
        """Why an Image block's file cannot be read (moved, its disk unplugged, damaged; translated), "" when it can or
        when none is chosen yet (that one is "choose its file"): its block is circled in red and Run refused."""
        src = getattr(p, "source", None)
        if not isinstance(src, dict) or "fusion" in src or not src.get("path"):
            return ""
        from ..stages.image import image_problem
        why = image_problem(self.p.abspath(src["path"]))
        return _(why).format(name=Path(str(src["path"])).name) if why else ""

    def data_problems(self, views) -> list[tuple[str, str]]:
        """(workflow name, reason) of the workflows among ``views`` whose data cannot be computed, said before the run
        (PRINCIPLES: nothing runs without data): a session's folder not there, a frame that cannot be read, calibration
        frames of another size than the lights (``session_problems``), fewer than 2 light frames to stack. From what the
        worker read (``Facts``): nothing is read from the disk here. A workflow without lights is not one of them
        (``has_data`` says it)."""
        out = []
        for q in views:
            if q.source is not None:                      # an Image (``image_problem``), a fusion (its channels)
                continue
            ses = [s for s in self.p.sessions_of(q.pid) if s.lights]
            if not ses:
                continue
            why = next((w for s in ses for w in self.session_problems(s.id)), "")
            if not why and all(s.id in self._facts.sessions for s in ses):
                blocks = {n.get("block") for n in self.graph(q) if not n.get("off")}
                total = sum(self._facts.sessions[s.id].get("lights", 0) for s in ses)
                gone = sum(1 for sid, _n in self.p.excluded if sid in {s.id for s in ses})
                if total == 0:
                    why = _("{session}: no light frame in its folders (camera raw, FITS, JPEG, PNG or TIFF files)").format(
                        session=self._session_name(ses[0].id))
                elif total - gone < 2 and blocks - {"calibration", "registration", "selection"}:
                    why = (_("only 1 light frame: stacking needs at least 2") if total == 1 else
                           _("{n} light frame kept of {total}: stacking needs at least 2").format(
                               n=max(total - gone, 0), total=total))
            if why:
                out.append((q.name, why))
        return out

    def _session_name(self, sid: int) -> str:
        try:
            return self.p.session(sid).name or _("Session {n}").format(n=sid)
        except (KeyError, ValueError, StopIteration):
            return _("Session {n}").format(n=sid)

    def session_problems(self, sid: int) -> list[str]:
        """Why a session's data cannot be computed, translated (``session_checks``, read by the worker): its block is
        circled in red and Run refused."""
        name = self._session_name(sid)
        return [_(t).format(session=name, **{k: _(v) if k == "kind" else v for k, v in a.items()})
                for t, a in self._facts.checks.get(sid) or []]

    def colour_note(self, sid: int) -> dict | None:
        """io.frames.colour_session of a session's lights (issue #57), as the worker last read them; None: mosaics."""
        return self._facts.colour.get(sid)

    def colour_frames(self, p: WorkflowView) -> bool:
        """A workflow that stacks frames already in colour (issue #57): no Bayer drizzle for it."""
        return p.source is None and any(self.colour_note(s) for s in self.sessions_of(p))

    def unreadable_lights(self, sid: int) -> int:
        """How many of a session's lights cannot be read (``session_checks``): they are not used."""
        return sum(int(a.get("unread", 0)) for _t, a in self._facts.checks.get(sid) or [] if a.get("frames") == "lights")

    def session_problem(self, sid: int) -> str:
        """The first of ``session_problems``, "" when there is none."""
        return next(iter(self.session_problems(sid)), "")

    def place_indices(self, p: WorkflowView, block: str) -> list[int]:
        """Where a block can be inserted in a workflow's graph."""
        if block == "color" and self.is_mono(p) or block == "stars" and is_fusion_view(p):
            return []
        return wfl.places(self.graph(p), self.source_kind(p), block)

    def insert_block(self, p: WorkflowView, block: str, index: int | None = None, method: str | None = None) -> str:
        """Place a block in a workflow's graph (``index``: where; None: its usual place). Returns the new node's
        view key; raises ValueError (translated) when it cannot go there."""
        g, src = self.graph(p), self.source_kind(p)
        if block == "color" and self.is_mono(p):
            raise ValueError(_(wfl.MONO_NO_COLOUR))
        if block == "stars" and is_fusion_view(p):
            raise ValueError(_(FUSION_STARS))
        if index is None:
            index = wfl.insert_place(g, src, block)
            if index is None:
                raise ValueError(self.palette_reason(p, block))
        e = wfl.place_problem(g, src, block, index)
        if e is not None:
            raise ValueError(refused_text(e))
        n = wfl.node(block, method, name=wfl.new_name(g, block))
        g.insert(index, n)
        with self._edit():
            self.p.set_blocks(p.pid, g)
            self._fresh_settings(p.pid, n)
        return node_key(n)

    def _fresh_settings(self, wid: int, n: dict):
        """A block placed from the palette starts from its defaults (question 9): not the settings a block of the same
        name left behind, nor those of the workflow a branch was copied from. Ctrl+Z gives them back."""
        w = self.p.workflow(wid)
        name = n["name"] if n["block"] in wfl.IMAGE_OPS else n["block"]
        try:
            stages = next((b.stages for b in api.blocks(self.p, wid) if b.block == name), ())
        except (ValueError, KeyError, OSError):
            stages = ()
        for st in stages:
            w.params.pop(st, None)

    def removal_problem(self, p: WorkflowView, key: str) -> str:
        """Why a block cannot be taken out of a workflow (translated), "" when it can."""
        i = self.node_index(p, key)
        if i is None:
            return _("not in workflow")
        e = wfl.remove_problem(self.graph(p), self.source_kind(p), i)
        if e is not None:
            return refused_text(e)
        g = self.graph(p)
        return self._fusion_need(p, g[:i] + g[i + 1:])    # a fusion's channel keeps what the fusion needs

    def remove_node(self, p: WorkflowView, key: str) -> None:
        """Take a block out of a workflow's graph; ValueError (translated) when the blocks after it need it."""
        why = self.removal_problem(p, key)
        if why:
            raise ValueError(why)
        g = self.graph(p)
        del g[self.node_index(p, key)]
        with self._edit():
            self.p.set_blocks(p.pid, g)

    def can_switch_off(self, p: WorkflowView, key: str) -> str:
        """Why a block cannot be switched off (translated), "" when it can."""
        i = self.node_index(p, key)
        if i is None:
            return _("not in workflow")
        e = wfl.off_problem(self.graph(p), self.source_kind(p), i)
        if e is not None:
            return refused_text(e)
        g = self.graph(p)
        return self._fusion_need(p, g[:i] + [dict(g[i], off=True)] + g[i + 1:])

    def set_graph(self, p: WorkflowView, blocks: list[dict]) -> None:
        """Give a workflow a whole graph (a template's)."""
        with self._edit():
            self.p.set_blocks(p.pid, blocks)

    def add_graph_workflow(self, name: str, blocks: list[dict], source: dict | None = None,
                           sessions: list[int] | None = None) -> WorkflowView:
        """A new workflow from a graph (issue #47): a template's, or an empty one that the palette builds."""
        with self._edit():
            q = self.add_path(self.unique_name(name), {"debruitage": False, "etoiles": "garder", "fond": "simple"})
            w = self.p.workflow(q.pid)
            w.source = source
            w.sessions = sessions
            self.p.set_blocks(w, blocks)
            self.p.validate()
        return self._view_of(q.pid)

    def ensure_frames_workflow(self) -> WorkflowView | None:
        """A project with sessions and no workflow from the frames gets an empty one (issue #47: a Session placed on a
        blank page starts a workflow, built from the palette)."""
        if not self.p.sessions or any(q.source is None for q in self.paths):
            return None
        return self.add_graph_workflow(_("Workflow"), [])

    def add_filter_session(self, flt: str) -> int:
        """An empty Session block for a filter (a template's, before its frames are known, issue #47)."""
        with self._edit():
            sid = max((s.id for s in self.p.sessions), default=0) + 1
            from ..core.project2 import Session
            name = api.unique_session_name(self.p, flt) if flt else self.new_session_name(sid)
            self.p.sessions.append(Session(sid, name, filter=flt))
            if len(self.p.sessions) >= 2 and self.p.stack_mode == "together":
                self.p.stack_mode = "per_session"
            self.p.validate()
        return sid

    def new_session_name(self, sid: int) -> str:
        """A new session's default name: "Session n", n its id or the next number no session's name has (GUI
        exploration 4: a session renamed "Session 2" by the user, the next one was "Session 2" too)."""
        taken = {s.name for s in self.p.sessions}
        n = sid
        while _("Session {n}").format(n=n) in taken:
            n += 1
        return _("Session {n}").format(n=n)

    def _nights_as_one(self, p: WorkflowView) -> bool:
        """Several nights of a workflow calibrated together (a project made before they could be apart)."""
        return p.source is None and len(self.sessions_of(p)) > 1 and not self.separate_sessions()

    def branch_keys(self, p: WorkflowView) -> list[str]:
        """The blocks a new branch can start after, in order: the Session (a Calibration in parallel), then the
        workflow's blocks; each night's Session when the nights are calibrated apart."""
        keys = list(self.display_keys(p))
        if p.source is not None:
            return keys
        sep = self.separate_sessions()
        if sep and any(n["block"] == "calibration" for n in self.graph(p)):
            keys.insert(0, "calibration")                 # drawn before the chain, one per night
        mine = self.sessions_of(p)
        return ([f"session@{sid}" for sid in sep if sid in mine] if sep else ["sources"]) + keys

    def _branch_night(self, p: WorkflowView, key: str) -> int | None:
        """The night a Calibration in parallel is placed on, from the Session block it follows."""
        if key == "sources":
            mine = self.sessions_of(p)
            return mine[0] if len(mine) == 1 else None
        return int(key.partition("@")[2]) if key.startswith("session@") else None

    def can_branch(self, p: WorkflowView, key: str, block: str) -> bool:
        """Whether ``block`` can start a new branch right after the block ``key`` of a workflow (issue #47); after a
        Session (``branch_keys``), a Calibration beside the workflow's own."""
        if key == "sources" or key.startswith("session@"):
            return block == "calibration" and p.source is None and self._branch_night(p, key) is not None and \
                any(n["block"] == "calibration" for n in self.graph(p))
        if block == "color" and self.is_mono(p):
            return False
        g = self.graph(p)
        i = self.node_index(p, key)
        if i is None or g[i]["off"]:
            return False
        head = g[:i + 1]
        return wfl.place_problem(head, self.source_kind(p), block, len(head)) is None

    def add_branch(self, p: WorkflowView, key: str, block: str) -> tuple[WorkflowView, str]:
        """A new branch (issue #47): a workflow with the same source, sessions and settings as ``p``, its graph up to
        the block ``key``, then ``block``. What they share is computed once (the same keys) and drawn once, as a fork.
        Returns the branch and its new block's key."""
        if not self.can_branch(p, key, block):
            raise ValueError(self.palette_reason(p, block) or _("no branch from here"))
        night = self._branch_night(p, key)
        g = self.graph(p)
        i = self.node_index(p, key)
        head = [dict(n) for n in g[:i + 1]] if night is None else []
        method = None
        nxt = g[i + 1] if i is not None and i + 1 < len(g) else None
        if nxt is not None and nxt["block"] == block and block in wfl.METHODS:
            # the same block with the same method would be the same result, drawn once: the branch tries another one
            method = next((m for m in wfl.METHODS[block] if m != nxt["method"]), None)
        n = wfl.node(block, method, name=wfl.new_name(head, block))
        head.append(n)
        flow = p.name.partition(" · ")[0]                 # beside a block of the frames: named from the flow it leaves,
        if block not in wfl.FRAME_BLOCKS or flow not in {w.name for w in self.p.workflows}:     # not the last branch
            flow = p.name
        with self._edit():                                # named by its block's kind, numbered as names are (not the
            # block's own "~2", which made "Ha · Gradient 2 2"; GUI exploration 4)
            w = api.add_workflow(self.p, self.unique_name(self.branch_name(flow, node_key(n))), None, None,
                                 copy_of=p.pid)
            x = self.p.workflow(w.id)
            x.source = copy.deepcopy(self.p.workflow(p.pid).source)
            x.sessions = copy.deepcopy(self.p.workflow(p.pid).sessions)
            x.end = None
            if night is not None:                         # beside the Calibration: this night's own, the others as
                x.calibration = {**(x.calibration or {}), str(night): {"fork": w.id}}      # in the workflow it left
            elif block in wfl.FRAME_BLOCKS:               # a frame block is drawn on its own, even while it computes
                head[-1]["fork"] = w.id                   # the same as the one beside it (computed once)
            self.p.set_blocks(w.id, head)
            self._fresh_settings(w.id, n)                 # its new block: from its defaults (question 9)
            self.p.validate()
        new = node_key(n)
        return self._view_of(w.id), f"{new}@{night}" if night is not None and self.separate_sessions() else new

    # ------------------------------------------------------------ a Calibration placed in parallel
    def calibration_fork(self, p: WorkflowView, sid: int) -> int | None:
        """What tells a Calibration placed in parallel on a night from the others; None: the session's calibration."""
        try:
            own = (self.p.workflow(p.pid).calibration or {}).get(str(sid))
        except KeyError:
            return None
        return own["fork"] if own else None

    def own_calibration_night(self, p: WorkflowView, sid: int | None = None) -> int | None:
        """The night of a workflow's Calibration placed in parallel (``sid``: that one, when it has one): None when
        the workflow calibrates as its sessions say."""
        try:
            own = self.p.workflow(p.pid).calibration or {}
        except KeyError:
            return None
        return next((int(n) for n in own if sid is None or int(n) == sid), None)

    def same_calibration(self, p: WorkflowView, sid: int) -> list[WorkflowView]:
        """The workflows going through a Calibration placed in parallel on a night ([] for a Session's own)."""
        mark = self.calibration_fork(p, sid)
        return [] if mark is None else [q for q in self.paths if self.calibration_fork(q, sid) == mark]

    def calibration_follows(self, p: WorkflowView, sid: int, kind: str) -> bool:
        """Whether a Calibration placed in parallel still takes a kind of frames from its Session."""
        return kind not in ((self.p.workflow(p.pid).calibration or {}).get(str(sid)) or {})

    def calibration_files(self, p: WorkflowView, sid: int, kind: str) -> list[str]:
        """The files of a kind a workflow's Calibration reads for a night (its own, else its Session's)."""
        if self.calibration_follows(p, sid, kind):
            return self.session_files(sid, kind)
        return [self.p.abspath(f) for f in self.p.workflow(p.pid).calibration[str(sid)][kind]]

    def _set_calibration_files(self, paths: list[WorkflowView], sid: int, kind: str, files: list[str] | None) -> None:
        with self._edit():
            for q in paths:
                own = self.p.workflow(q.pid).calibration[str(sid)]
                if files is None:
                    own.pop(kind, None)
                else:
                    own[kind] = list(files)
            self.p.validate()

    def add_calibration_files(self, paths: list[WorkflowView], sid: int, kind: str, files) -> int:
        """Files added to a Calibration placed in parallel (``paths``: the workflows going through it): the kind stops
        following its Session. Returns how many were new."""
        cur = self.calibration_files(paths[0], sid, kind)
        have = {norm_path(f) for f in cur}
        new = [Path(f).as_posix() for f in files if norm_path(f) not in have]
        if new:
            self._set_calibration_files(paths, sid, kind, cur + new)
        return len(new)

    def remove_calibration_files(self, paths: list[WorkflowView], sid: int, kind: str, files) -> bool:
        """Files taken out of a Calibration placed in parallel (the files themselves are never touched)."""
        cur = self.calibration_files(paths[0], sid, kind)
        gone = {norm_path(f) for f in files}
        keep = [f for f in cur if norm_path(f) not in gone]
        if len(keep) == len(cur):
            return False
        self._set_calibration_files(paths, sid, kind, keep)
        return True

    def follow_session(self, paths: list[WorkflowView], sid: int, kind: str) -> None:
        """A kind of a Calibration placed in parallel given back to its Session."""
        self._set_calibration_files(paths, sid, kind, None)

    def remove_own_calibration(self, p: WorkflowView, sid: int) -> None:
        """A Calibration placed in parallel taken out: the workflow calibrates that night as its Session says again."""
        with self._edit():
            w = self.p.workflow(p.pid)
            own = dict(w.calibration or {})
            own.pop(str(sid), None)
            w.calibration = own or None

    def _forks(self, q: WorkflowView) -> tuple:
        """What sets a workflow's Calibrations placed in parallel apart: its nights' and each one's mark."""
        try:
            own = self.p.workflow(q.pid).calibration or {}
        except KeyError:
            return ()
        return tuple(sorted((int(n), e["fork"]) for n, e in own.items()))

    def can_send_to_post(self, p: WorkflowView, key: str) -> bool:
        """Whether a block's image can be sent to Post as a layer of its own (issue #47): a block giving an image, not
        the last one (its image goes to Post already), not switched off, and not sent yet."""
        if p.source is not None and "fusion" in p.source:
            return False
        g = self.graph(p)                                 # a workflow of options too: its graph, the same keys
        i = self.node_index(p, key)
        if i is None or g[i]["off"] or g[i]["block"] not in wfl.IO or wfl.IO[g[i]["block"]][1] != wfl.IMAGE:
            return False
        if i == len(g) - 1 or all(n.get("off") for n in g[i + 1:]):
            return False
        head = [(node_key(n), n.get("fork")) for n in g[:i + 1]]
        return not any(q.pid != p.pid and q.source == p.source and self.sessions_of(q) == self.sessions_of(p)
                       and self._forks(q) == self._forks(p)
                       and [(node_key(n), n.get("fork")) for n in self.graph(q)] == head for q in self.paths)

    def is_sent_branch(self, p: WorkflowView) -> bool:
        """Whether a workflow is only a block's image sent to Post: its blocks are the first blocks of another workflow
        of the same source, nothing of its own after them (question 1: its wire taken away, it goes)."""
        def nodes(q):                                     # a block with its method: another method is its own
            return [(node_key(n), n.get("method"), bool(n.get("off")), n.get("fork")) for n in self.graph(q)
                    if n["block"] != "output"]
        mine = nodes(p)
        if not mine or not self.is_graph(p):
            return False
        return any(q.pid != p.pid and q.source == p.source and self.sessions_of(q) == self.sessions_of(p)
                   and self._forks(q) == self._forks(p)
                   and nodes(q)[:len(mine)] == mine and len(nodes(q)) > len(mine) for q in self.paths)

    def is_emptied_branch(self, p: WorkflowView) -> bool:
        """Whether a branch has no block of its own any more: it is only the first blocks of an older workflow (its
        trunk), or all of them. Its last own block deleted, it goes with it (control round: it stayed, sending the
        trunk's image). A branch is named after its trunk ("Workflow · Color", as made and as its name follows): two
        workflows of their own (two templates) are never one's branch."""
        def nodes(q):
            return [(node_key(n), n.get("method"), bool(n.get("off")), n.get("fork")) for n in self.graph(q)
                    if n["block"] != "output"]
        mine = nodes(p)
        if not mine:
            return False
        return any(q.pid < p.pid and p.name.startswith(q.name + " · ") and q.source == p.source
                   and self.sessions_of(q) == self.sessions_of(p) and self._forks(q) == self._forks(p)
                   and nodes(q)[:len(mine)] == mine for q in self.paths)

    def send_to_post(self, p: WorkflowView, key: str, name: str) -> WorkflowView:
        """A block's image as a layer of Post (issue #47): a branch of ``p`` up to that block, its Output its own;
        what they share is computed once."""
        if not self.can_send_to_post(p, key):
            raise ValueError(_("no Post from here"))
        head = [dict(n) for n in self.graph(p)[:self.node_index(p, key) + 1]]
        with self._edit():
            w = api.add_workflow(self.p, self.unique_name(name), None, None, copy_of=p.pid)
            x = self.p.workflow(w.id)
            x.source = copy.deepcopy(self.p.workflow(p.pid).source)
            x.sessions = copy.deepcopy(self.p.workflow(p.pid).sessions)
            x.end = None
            self.p.set_blocks(w.id, head)
            self.p.validate()
        return self._view_of(w.id)

    def add_parallel_flow(self) -> WorkflowView:
        """A second flow of frames (issue #47: a Session dropped on the new-flow zone): its own empty Session, the
        flows that took every session keep the ones they had."""
        with self._edit():
            mine = [s.id for s in self.p.sessions]
            frames = [w for w in self.p.workflows if w.source is None and w.sessions is None]
            if frames and not mine:                       # a template's flow, no night yet (1.5): its own Session first
                mine = [self.add_filter_session("")]
            for w in frames:
                w.sessions = list(mine)                   # they keep their nights
            sid = self.add_filter_session("")
            q = self.add_graph_workflow(_("Flow"), [], sessions=[sid])
        return q

    def _sync_graphs(self):
        """The graphs follow options set by code that speaks options (a method chosen in the panel...)."""
        for w in self.p.workflows:
            if w.blocks is None:
                continue
            src = wfl.source_of(w)
            try:
                o = wfl.check_options(w.options)
                if wfl.options_of(w.blocks, src) != o:
                    self.p.set_blocks(w, wfl.sync_blocks(w.blocks, src, o))
            except ValueError:
                continue

    def is_mono(self, p: WorkflowView) -> bool:
        """A workflow of monochrome frames: each of its sessions names a filter."""
        try:
            return p.source is None and self.p.is_mono(p.pid)
        except KeyError:
            return False

    def sessions_of(self, p: WorkflowView) -> list[int]:
        """The ids of the sessions a workflow stacks."""
        return [s.id for s in self.p.sessions_of(p.pid)] if p.source is None else []

    def filters(self) -> dict[str, list[int]]:
        """{filter: its sessions' ids} of the project's monochrome sessions."""
        out: dict[str, list[int]] = {}
        for s in self.p.sessions:
            if s.filter:
                out.setdefault(s.filter, []).append(s.id)
        return out

    def filter_workflow(self, flt: str, opts: dict | None = None, name: str | None = None) -> WorkflowView | None:
        """The workflow stacking this filter's sessions (made when there is none). None without such sessions."""
        sids = self.filters().get(flt)
        if not sids:
            return None
        for q in self.paths:
            if q.source is None and q.sessions is not None and sorted(q.sessions) == sorted(sids) and not q.end:
                return q
        o = dict(opts or {"debruitage": True, "etoiles": "garder", "fond": "simple"})
        with self._edit():
            if self.p.stack_mode == "together":
                self.p.stack_mode = "per_session"
            w = api.add_workflow(self.p, self.unique_name(name or flt), wfl.check_options(to_engine(o)), {})
            self.p.workflow(w.id).sessions = list(sids)
            self.p.validate()
        return self._view_of(w.id)

    def add_fusion(self, name: str, palette: str, inputs: list[dict], stars: dict | None = None) -> WorkflowView:
        """A fusion workflow (issue #43): the outputs of ``inputs`` [{"pid", "role", "channel"}] combined with a
        palette, in the Composition section; its Output is a layer of Post. ``stars`` (option A): the channels'
        starless images fused, and the star layer's source ({"mode": "palette" | "channel" + "role" | "workflow" +
        "pid"}); the channels' workflows get Stars/background."""
        with self._edit():
            o = {"debruitage": bool(stars), "etoiles": "retirer" if stars else "garder", "fond": "simple"}
            w = api.add_workflow(self.p, self.unique_name(name), wfl.check_options(to_engine(o)), {})
            f = {"palette": palette, "inputs": [
                {"pid": int(i["pid"]), "role": str(i["role"]), "channel": str(i.get("channel", "L"))} for i in inputs]}
            if stars:
                f["stars"] = dict(stars)
                added = self._with_stars([i["pid"] for i in inputs] + ([stars["pid"]] if stars.get("pid") else []))
                if added:                                 # what it added, taken back when it goes (decision 2)
                    f["added"] = added
            self.p.workflow(w.id).source = {"fusion": f}
            self.p.validate()
        return self._view_of(w.id)

    def _with_stars(self, pids) -> list[dict]:
        """These workflows get Stars/background (and Denoise, which it needs). Returns what each changed one had
        before: [{"pid", "denoise", "stars"}] (the fusion keeps it, the engine does not read it)."""
        out = []
        for pid in dict.fromkeys(pids):
            w = self.p.workflow(int(pid))
            if not w.options.get("stars"):
                out.append({"pid": int(pid), "denoise": w.options.get("denoise"), "stars": w.options.get("stars")})
                w.options = wfl.check_options({**w.options, "denoise": w.options.get("denoise") or "pretrained",
                                               "stars": "neural"})
        return out

    def remove_fusion(self, p: WorkflowView):
        """A Combine channels taken out (GUI exploration 2, decision 2): its workflow, and what placing it added to its channels
        (their Denoise and Stars/background), unless another fusion still takes their starless images: the channels
        come back as they were before it."""
        f = (p.source or {}).get("fusion") or {}
        with self._edit():
            api.delete_workflow(self.p, p.pid)
            still = {}                                    # channel id: a fusion left that takes its starless image
            for w in self.p.workflows:
                g = (w.source or {}).get("fusion") if isinstance(w.source, dict) else None
                if g and g.get("stars"):
                    for pid in [i["pid"] for i in g["inputs"]] + ([g["stars"]["pid"]] if g["stars"].get("pid") else []):
                        still.setdefault(pid, g)
            for a in f.get("added") or []:
                if a.get("pid") in still:                 # that fusion takes it back when it goes
                    g = still[a["pid"]]
                    g["added"] = list(g.get("added") or []) + [a]
                    continue
                if a.get("pid") not in {w.id for w in self.p.workflows}:
                    continue
                w = self.p.workflow(a["pid"])
                w.options = wfl.check_options({**w.options, "denoise": a.get("denoise"), "stars": a.get("stars")})
            self.p.validate()
        self.current = min(self.current, len(self.paths) - 1)
        self.currentChanged.emit()

    def set_fusion_weight(self, p: WorkflowView, role: str, weight: float) -> bool:
        """A channel's weight in a fusion (1: as balanced)."""
        w = self.p.workflow(p.pid)
        cur = dict(w.source["fusion"].get("weights") or {})
        weight = round(float(weight), 3)
        if cur.get(role, 1.0) == weight:
            return False
        with self._edit():
            ws = api.workflows(self.p)
            x = next(v for v in ws if v.id == p.pid)
            cur[role] = weight
            x.source = {"fusion": {**x.source["fusion"], "weights": {k: v for k, v in cur.items() if v != 1.0}}}
            api.set_workflows(self.p, ws, self.p.current)
        return True

    def _source_refused(self, q) -> str:
        """Why a workflow cannot feed a fusion, as a channel or as its star source ("" when it can; issue #55)."""
        return _("An image already stretched: a stretch cannot be undone faithfully") \
            if self.source_kind(q) == "stretched" else ""

    def star_choices(self, p: WorkflowView) -> list[tuple[WorkflowView, str]]:
        """The workflows the stars of the fusion ``p`` can come from, each with why it cannot be chosen ("" when it can;
        issue #55): the colour stacks and the linear images; an image already stretched is listed greyed."""
        return [(q, self._source_refused(q)) for q in self.paths
                if q.pid != p.pid and not is_fusion_view(q) and not self.is_mono(q)]

    def channel_choices(self, p: WorkflowView, role: str) -> list[tuple[WorkflowView, str]]:
        """The workflows a channel of the fusion ``p`` can read, each with why it cannot be chosen now ("" when it can;
        issue #55): the stacks of its filter and the linear images; an image already stretched is listed greyed; never
        a fusion or a workflow that ends early."""
        f = p.source["fusion"]
        i = next(x for x in f["inputs"] if x["role"] == role)
        out = []
        for q in self.paths:
            if q.pid != i["pid"]:
                if q.pid == p.pid or is_fusion_view(q) or q.end:
                    continue
                why = self._channel_problem(q, {**i, "pid": q.pid}, role, f, False)
                if why is not None and "filter" in dict(why[1]):
                    continue
            out.append((q, self._source_refused(q)))
        return out

    def set_fusion_channel(self, p: WorkflowView, role: str, pid: int) -> bool:
        """The workflow a channel of a fusion reads (issue #55), its colour kept. A starless fusion gives it its
        Stars/background, as when it is placed."""
        f = self.p.workflow(p.pid).source["fusion"]
        if not is_fusion_view(p) or next(i for i in f["inputs"] if i["role"] == role)["pid"] == int(pid) or \
                self._source_refused(self.view(int(pid))):
            return False
        with self._edit():
            ws = api.workflows(self.p)
            x = next(v for v in ws if v.id == p.pid)
            x.source = {"fusion": {**x.source["fusion"], "inputs": [
                {**i, "pid": int(pid)} if i["role"] == role else i for i in x.source["fusion"]["inputs"]]}}
            api.set_workflows(self.p, ws, self.p.current)
            added = self._with_stars([int(pid)]) if f.get("stars") else []
            if added:                                     # taken back when the fusion goes (decision 2)
                g = self.p.workflow(p.pid).source["fusion"]
                g["added"] = list(g.get("added") or []) + added
                self.p.validate()
        return True

    def set_fusion_stars(self, p: WorkflowView, stars: dict | None) -> bool:
        """The star source of a fusion (option A), or None: the channels combined with their stars."""
        w = self.p.workflow(p.pid)
        if not is_fusion_view(p) or (w.source["fusion"].get("stars") or None) == (stars or None) or \
                ((stars or {}).get("pid") and self._source_refused(self.view(stars["pid"]))):
            return False
        with self._edit():
            ws = api.workflows(self.p)
            x = next(v for v in ws if v.id == p.pid)
            f = {k: v for k, v in x.source["fusion"].items() if k != "stars"}
            if stars:
                f["stars"] = dict(stars)
            x.source = {"fusion": f}
            x.options = wfl.check_options({**x.options, "denoise": "pretrained" if stars else None,
                                           "stars": "neural" if stars else None})
            api.set_workflows(self.p, ws, self.p.current)
            if stars:
                added = self._with_stars([i["pid"] for i in f["inputs"]] + ([stars["pid"]] if stars.get("pid") else []))
                if added:                                 # taken back when the fusion goes (decision 2)
                    g = self.p.workflow(p.pid).source["fusion"]
                    g["added"] = list(g.get("added") or []) + added
                self.p.validate()
        return True

    def set_fusion_palette(self, p: WorkflowView, palette: str) -> bool:
        """Another palette for a fusion (its channels kept when the new one reads the same roles). Its name follows
        while it is the one its palette gave it ("SHO (Hubble palette)" → "HSO"); a name the user gave stays (GUI
        exploration 4: an HSO fusion kept saying SHO)."""
        w = self.p.workflow(p.pid)
        if not is_fusion_view(p) or w.source["fusion"]["palette"] == palette:
            return False
        from ..texts import choice_label
        from .gallery import GALLERY
        from .i18n import language

        def given(pal):                                   # the names a palette gives: its own, its templates'
            names = {pal, choice_label(pal, "en")} | {t["name"] for t in GALLERY if t.get("palette") == pal}
            return names | {_(x) for x in names}
        base, _sp, num = w.name.rpartition(" ")           # "HSO 2": a second one, numbered by unique_name
        follow = w.name in given(w.source["fusion"]["palette"]) or \
            (num.isdigit() and base in given(w.source["fusion"]["palette"]))
        with self._edit():
            ws = api.workflows(self.p)
            x = next(v for v in ws if v.id == p.pid)
            x.source = {"fusion": {**x.source["fusion"], "palette": palette}}
            api.set_workflows(self.p, ws, self.p.current)
            if follow:
                self._rename(p.pid, self.unique_name(choice_label(palette, language())))
        return True

    def set_node_method(self, paths, key: str, method: str) -> bool:
        """The method of a block placed twice ("fond~2", the engine's ``method``): its own, stored on its node (the
        workflow's options set the first one only). One change, for Undo; False when nothing changes."""
        changed = False
        with self._edit():
            for q in paths:
                g = self.graph(q)
                i = self.node_index(q, key)
                if i is None or g[i].get("method") == method:
                    continue
                g[i] = dict(g[i], method=method)
                self.p.set_blocks(q.pid, g)
                changed = True
        return changed

    def is_cut(self, p: WorkflowView, key: str) -> bool:
        """Whether the user switched this block off (issue #47)."""
        return any(n["off"] for n in self.graph(p) if node_key(n) == key)

    def greyed_reason(self, p: WorkflowView, key: str) -> str:
        """Why this block of the workflow cannot run (translated), or ""."""
        why = self.greyed(p).get(engine_block(key), "")
        return _(why) if why else ""

    def set_cut(self, paths, key: str, on: bool) -> bool:
        """Switch a block off (``on``) in these workflows, or on again (issue #47): one change, for Undo. Its settings
        and results are kept; switched off, the data passes through it. False when nothing changes; ValueError
        (translated) when it cannot be switched off."""
        paths = [q for q in paths if self.is_cut(q, key) != on]
        if not paths:
            return False
        for q in paths:
            why = self.can_switch_off(q, key) if on else ""
            if why:
                raise ValueError(why)
        with self._edit():
            for q in paths:
                g = self.graph(q)
                i = self.node_index(q, key)
                g[i] = dict(g[i], off=on)
                e = wfl.graph_problem(wfl.with_output(g, self.source_kind(q)), self.source_kind(q))
                if e is not None:
                    raise ValueError(refused_text(e))
                self.p.set_blocks(q.pid, g)
        return True

    def first_to_fill(self, p: WorkflowView) -> tuple[WorkflowView, str] | None:
        """(workflow, block) of the first block of ``p`` that waits for the user: a Session without lights, an Image
        without a file; for a Combine channels, its channels'. None when nothing waits (a new project from a template opens
        there, GUI exploration 2)."""
        src = p.source or {}
        if "fusion" in src:
            for i in src["fusion"].get("inputs", []):
                q = self.view(int(i["pid"]))
                got = self.first_to_fill(q) if q is not None and q is not p else None
                if got is not None:
                    return got
            return None
        if p.source is not None:
            return None if src.get("path") else (p, "image")
        sep = self.separate_sessions()
        for s in self.p.sessions_of(p.pid):
            if not s.lights:
                return p, (f"session@{s.id}" if s.id in sep else "sources")
        return None

    def separate_sessions(self) -> list[int]:
        """The ids of the sessions calibrated each on its own, drawn as Session and Calibration blocks before the
        chain (two or more, in the per-session mode); [] otherwise."""
        return self.p.separate_sessions()

    def session(self, sid: int):
        return self.p.session(sid)

    def add_session(self, name: str = "", lights=(), **calibration) -> int | None:
        """A new session (issue #43): its lights, and its calibration frames by kind (``darks``, ``flats``, ``bias``,
        ``flat_darks``): folders or files. A project made before issue #43 (all its nights calibrated together) keeps
        that mode, so the new session joins the others' calibration. Returns its id."""
        from ..core.project2 import Session
        bad = set(calibration) - set(CALIBRATION_KINDS)
        if bad:
            raise ValueError(f"not kinds of calibration frames: {sorted(bad)}")
        sid = max((s.id for s in self.p.sessions), default=0) + 1
        with self._edit():
            self.p.sessions.append(Session(sid, api.unique_session_name(self.p, name) if name else
                                           self.new_session_name(sid),
                                           lights=[Path(x).as_posix() for x in lights],
                                           **{k: [Path(x).as_posix() for x in v] for k, v in calibration.items()}))
            self.p.validate()
        self.reload_sources()
        return sid

    def flows_of_session(self, sid: int) -> list[WorkflowView]:
        """The workflows that read this session alone: they go with it (issue #47). The last session: all those from the
        frames."""
        last = len(self.p.sessions) <= 1
        return [q for q in self.paths if q.source is None and (q.sessions is None and last or q.sessions is not None
                                                               and list(q.sessions) == [sid])]

    def session_removal_problem(self, sid: int) -> str:
        """Why a session cannot be taken out (translated), "" when it can: the only session of a fusion's channel stays
        (GUI exploration 2, decision 3: no broken project)."""
        for q in self.flows_of_session(sid):
            why = self.channel_problem(q)
            if why:
                return why
        return ""

    def remove_session(self, sid: int) -> bool:
        """Take a session out of the project (its files are never touched); the only one of a fusion's channel stays
        (``session_removal_problem``). The last one leaves a project without frames: its workflows from them go too."""
        if self.session_removal_problem(sid):
            return False
        with self._edit():
            self.p.sessions = [s for s in self.p.sessions if s.id != sid]
            self.p.excluded = [x for x in self.p.excluded if x[0] != sid]
            for w in self.p.workflows:
                for k in [k for k in w.params if k.endswith(f"@{sid}")]:
                    del w.params[k]
                if w.calibration and str(sid) in w.calibration:    # its Calibration placed in parallel on that night
                    w.calibration = {n: e for n, e in w.calibration.items() if n != str(sid)} or None
            for w in list(self.p.workflows):              # a flow of that session alone goes with it (issue #47)
                if w.source is None and w.sessions is None and not self.p.sessions:
                    api.delete_workflow(self.p, w.id)
                elif w.source is None and w.sessions is not None and sid in w.sessions:
                    w.sessions = [x for x in w.sessions if x != sid]
                    if not w.sessions:
                        api.delete_workflow(self.p, w.id)
            self.p.validate()
        self.current = max(0, min(self.current, len(self.paths) - 1))
        self.reload_sources()
        return True

    def name_problem(self, name: str, pid: int | None = None, sid: int | None = None) -> str:
        """Why a workflow (``pid``) or a session (``sid``) cannot take this name (translated), "" when it can (GUI
        exploration 2, decision 8): never empty, never the name of another workflow, or of another session."""
        name = name.strip()
        if not name:
            return _("a name is needed")
        if sid is not None and any(s.name == name for s in self.p.sessions if s.id != sid):
            return _("“{name}”: another session has it").format(name=name)
        if pid is not None and any(w.name == name for w in self.p.workflows if w.id != pid):
            return _("“{name}”: another workflow has it").format(name=name)
        return ""

    def rename_session(self, sid: int, name: str) -> bool:
        """A session's new name; False when it does not change or is refused (``name_problem`` says why)."""
        name = name.strip()
        s = self.p.session(sid)
        if name == s.name or self.name_problem(name, sid=sid):
            return False
        with self._edit():
            s.name = name
        return True

    def set_session_filter(self, sid: int, flt: str) -> bool:
        """The filter of a session's lights (mono is planned: stored only)."""
        s = self.p.session(sid)
        if flt == s.filter:
            return False
        with self._edit():
            s.filter = flt
            self.p.validate()
        return True

    def new_session(self, pid: int | None = None) -> int:
        """A new empty session in the project ("+ Session", Add session…): it joins the flow ``pid`` (else the one shown),
        or starts one on a blank page; its frames are added in its Inspect. A project of one session calibrates each
        session on its own from then on; one made before with several nights calibrated together keeps that."""
        with self.batch():
            if self.stack_mode == "together" and len(self.p.sessions) == 1:
                self.set_stack_mode("per_session")
            sid = self.add_session()
            made = self.ensure_frames_workflow()
            q = made or self.join_session(self.view(pid) if pid is not None else self.path, sid)
        if q is not None:
            self.current = self.paths.index(self.view(q.pid))
        return sid

    def add_session_entries(self, sid: int, kind: str, paths, notes: list | None = None) -> int:
        """Files added to a session's list of a kind (lights, darks, flats, bias, flat_darks): a folder given is a
        shortcut for its frames, listed once and for all. Returns how many files were new. GUI exploration 7.4, 7.5: a
        file already there, or in a folder a project made before lists, is already in the session; a folder without
        frames adds nothing; the lights' filter, when the session has none, is read from their FITS header. ``notes``:
        what was not added and why, (English text, the file's name) to say."""
        from ..core.project import frame_kind, list_frames
        if kind not in ("lights",) + tuple(CALIBRATION_KINDS):
            raise ValueError(f"not a kind of frames: {kind!r}")
        s = self.p.session(sid)
        notes = [] if notes is None else notes
        cur = list(getattr(s, kind))
        have = {norm_path(self.p.abspath(e)) for e in cur}
        folders = {norm_path(self.p.abspath(e)) for e in cur if Path(self.p.abspath(e)).is_dir()}

        def known(f) -> bool:
            return norm_path(f) in have or (bool(frame_kind(Path(f))) and norm_path(str(Path(f).parent)) in folders)
        new: list[str] = []
        for x in paths:
            px = Path(x)
            files = [str(f) for f in list_frames([str(px)])] if px.is_dir() else [str(px)]
            if not files:
                notes.append(("“{name}” holds no frames (camera raw, FITS, JPEG, PNG or TIFF): not added.", px.name))
            elif all(known(f) for f in files):
                notes.append(("“{name}” is already in the session.", px.name))
            for f in files:
                if not known(f):
                    new.append(Path(f).as_posix())
                    have.add(norm_path(f))
        if not new:
            return 0
        flt = ""
        if kind == "lights" and (not s.filter or not cur):
            # a session's first lights say its filter, over the one it took from the flow it joined (an OIII night
            # added to an H-alpha flow is not stacked with the H-alpha frames)
            from ..io.frames import frame_filter
            first = next((f for f in list_frames(new) if frame_kind(f) == "fits"), None)
            flt = frame_filter(first) if first is not None else ""
        moved = bool(flt) and bool(s.filter) and flt != s.filter
        with self._edit():
            setattr(s, kind, cur + new)
            self._note_left_out(s, kind, new, False)      # a frame left out by hand, given again: in again
            if flt:
                s.filter = flt
            if moved:
                self._refile_session(sid)
            self.p.validate()
        self.reload_sources()
        return len(new)

    def _refile_session(self, sid: int) -> None:
        """A session whose filter changed leaves the flow it was in and joins its filter's (``join_session``)."""
        frames = [w for w in self.p.workflows if w.source is None]
        w = next((x for x in frames if x.sessions is not None and sid in x.sessions), None) \
            or next((x for x in frames if x.sessions is None), None)
        if w is None:
            return
        for x in frames:
            if x.sessions is not None and sid in x.sessions:
                x.sessions = [k for k in x.sessions if k != sid]
        self.join_session(self._view_of(w.id), sid)

    def session_files(self, sid: int, kind: str) -> list[str]:
        """The files a session's list gives for a kind (its folders' frames, its files), as the worker last read them."""
        return list((self._facts.files.get(sid) or {}).get(kind) or [])

    def file_warnings(self, sid: int) -> list[dict]:
        """What may be a mistake among a session's files (issue #58): core.project.type_warnings."""
        from ..core.project import type_warnings
        return type_warnings(self._facts.files.get(sid) or {}, self._facts.masters.get(sid) or set())

    def remove_session_files(self, sid: int, kind: str, files: list[str]) -> bool:
        """Files taken out of a session (issue #58; the files themselves are never touched): a file chosen alone leaves
        the list; a file of a folder chosen whole turns that folder into its other files. The last light can go too
        (the session then waits for its lights)."""
        s = self.p.session(sid)
        gone = {norm_path(self.p.abspath(f)) for f in files}
        keep: list[str] = []
        for e in getattr(s, kind):
            pth = self.p.abspath(e)
            if norm_path(pth) in gone:
                continue
            if not Path(e).suffix:                        # a folder: its frames as the worker read them
                inside = [x for x in self.session_files(sid, kind) if norm_path(str(Path(x).parent)) == norm_path(pth)]
                if any(norm_path(x) in gone for x in inside):
                    keep += [x for x in inside if norm_path(x) not in gone]
                    continue
            keep.append(e)
        once: dict[str, str] = {}                         # a file of a folder that was also chosen alone: once (7.4)
        for e in keep:
            once.setdefault(norm_path(self.p.abspath(e)), e)
        keep = list(once.values())
        if keep == list(getattr(s, kind)):
            return False
        with self._edit():
            setattr(s, kind, keep)
            self.p.validate()
        self.reload_sources()
        return True

    def remove_session_entry(self, sid: int, kind: str, entry: str) -> bool:
        """A folder or file taken out of a session's list (the files themselves are never touched); the last light
        too (issue #58: the session then waits for its lights)."""
        s = self.p.session(sid)
        lst = list(getattr(s, kind))
        keep = [e for e in lst if norm_path(self.p.abspath(e)) != norm_path(self.p.abspath(entry))]
        if len(keep) == len(lst):
            return False
        with self._edit():
            setattr(s, kind, keep)
            self.p.validate()
        self.reload_sources()
        return True

    def use_from_session(self, src: int, dst: int, kind: str) -> bool:
        """A session takes another session's calibration frames of a kind (the same files: the master is computed
        once for both)."""
        if kind not in CALIBRATION_KINDS:
            raise ValueError(f"not a kind of calibration frames: {kind!r}")
        frm, to = self.p.session(src), self.p.session(dst)
        want = [self.p.abspath(e) for e in getattr(frm, kind)]
        if [self.p.abspath(e) for e in getattr(to, kind)] == want:
            return False
        with self._edit():
            setattr(to, kind, [Path(x).as_posix() for x in want])
            self.p.validate()
        self.reload_sources()
        return True

    def shared_with(self, sid: int, kind: str) -> list[int]:
        """The other sessions that use the same calibration frames of a kind as this one."""
        mine = sorted(norm_path(self.p.abspath(e)) for e in getattr(self.p.session(sid), kind))
        if not mine:
            return []
        return [o.id for o in self.p.sessions if o.id != sid
                and sorted(norm_path(self.p.abspath(e)) for e in getattr(o, kind)) == mine]

    def add_image_workflow(self, name: str, path: str, opts: dict | None = None, stretched: bool = False
                           ) -> WorkflowView:
        """A workflow that starts from an image file (issue #43): a linear stack, with the blocks after the stack; or
        an image already stretched, to Post as it is (or its stars taken off first)."""
        o = dict(opts or {"debruitage": False, "etoiles": "garder", "fond": "simple"})
        if stretched and o.get("etoiles") == "retirer":
            o["debruitage"] = True                     # the options' rule; a stretched image's chain reads only Stars
        with self._edit():
            w = api.add_workflow(self.p, self.unique_name(name), wfl.check_options(to_engine(o)), {})
            self.p.workflow(w.id).source = {"path": Path(path).as_posix() if path else "", "stretched": bool(stretched)}
            self.p.validate()
        return self._view_of(w.id)

    def set_image_file(self, p: WorkflowView, path: str | None = None, stretched: bool | None = None):
        """The file of an Image block, or its kind (linear, or stretched already): chosen in its panel (an Image
        placed from the palette starts without one)."""
        from ..stages.image import stretched_by_nature
        src = dict(self.p.workflow(p.pid).source or {})
        before = stretched_by_nature(src.get("path") or "")
        if path is not None:
            src["path"] = Path(path).as_posix() if path else ""
        if stretched is not None:
            src["stretched"] = bool(stretched)
        elif path and stretched_by_nature(path) and not self.graph(p):
            src["stretched"] = True        # a PNG/JPEG or 8-bit TIFF is stretched by nature; still changeable (1.7)
        elif path and before and not self.graph(p):
            src["stretched"] = False       # the kind such a picture gave goes with it: a FITS or TIFF after it is linear
        with self._edit():
            w = self.p.workflow(p.pid)
            blocks = w.blocks
            w.source = src
            if blocks is not None:                        # the kind decides the Output (a stretched image has none)
                self.p.set_blocks(w.id, [n for n in blocks if n["block"] != "output"])
            self.p.validate()

    def block_key(self, p: WorkflowView, key: str):
        """What two workflows must have equal to share a block (with everything before it): the keys of its stages
        (the engine's ``blocks``); Selection is the thresholds (it has no stage of its own); Output has the
        workflow's id."""
        # a Calibration or another frame block placed in parallel is a block of its own, and what follows it too, even
        # while they compute the same (one result): the user set it there
        own = (self.node_fork(p, key), self._forks(p))
        if key == "align_images":                          # one per image, drawn together
            return ("align_images", p.pid)
        if key == "crop":
            zone = (p.params.get("stack") or {}).get("zone")
            return ("crop", tuple(zone) if zone else None) + own
        if key == "selection":
            st = p.params.get("stack") or {}
            base = ("selection", st.get("min_transparency"), st.get("max_fwhm_ratio"),
                    st.get("selection", "min_weight"), st.get("selection_transparency", "on"))
            return base + own if own != (None, ()) else base
        ref = next((b for b in self._blocks.get(p.pid, []) if view_key(b.block) == key), None)
        # GUI exploration 2: flows of other sessions are never one chain, even before their frames give them other
        # keys (a filters template's); two Image blocks without a file are two blocks (decision 4); a second Combine channels
        # is its own block (decision 2)
        mine = p.pid if is_fusion_view(p) or (isinstance(p.source, dict) and "path" in p.source
                                             and not p.source.get("path")) else None
        base = (key, ref.key if ref is not None else (), tuple(sorted(self.sessions_of(p))), mine)
        return base + own if own != (None, ()) else base

    def node_fork(self, p: WorkflowView, key: str) -> int | None:
        """The mark of a frame block placed in parallel (``wfl.node``), None for the others."""
        return next((n.get("fork") for n in self.graph(p) if node_key(n) == key), None)

    def title(self, stage: str) -> str:
        names = {"inventory": "Inventory", "calibrate": "Calibration", "register": "Registration",
                 "stack": "Stacking", "astrometry": "Astrometry", "reference": "Reference image",
                 "catalogs": "Gaia catalogues", "surveys": "H-alpha survey", "color": "Color", "denoise": "Denoise",
                 "gradient": "Sky gradient", "render_simple": "Output prep", "star_removal": "Star removal",
                 "deconv": "Deconvolution", "synth_stars": "Synthetic stars"}
        b, sep, n = stage.partition("~")                  # a graph's later block (issue #47): "Denoise 2"
        return _(names.get(b, b)) + (f" {n}" if sep else "")

    def state(self, p: WorkflowView | None, stage: str) -> str:
        """done | pending (never computed) | stale (computed once, a setting changed since) | running | failed |
        interrupted | skipped."""
        p = p or self.path
        st = self._states.get((p.pid, stage)) if p is not None else None
        return st.state if st is not None else "pending"

    def state_info(self, p: WorkflowView | None, stage: str):
        """The API's StageState (state, reason, key, previous key) of a stage, or None."""
        p = p or self.path
        return self._states.get((p.pid, stage)) if p is not None else None

    def previous_key(self, p: WorkflowView | None, stage: str) -> str | None:
        st = self.state_info(p, stage)
        return st.previous_key if st is not None else None

    def stage_key(self, p: WorkflowView, stage: str):
        """Two workflows compute a stage the same way when this key is equal (the engine's, from the definition)."""
        st = self._states.get((p.pid, stage))
        return st.key if st is not None and st.key else (stage, p.pid)

    def sharing(self, p: WorkflowView, stage: str) -> list[WorkflowView]:
        """The workflows that compute this stage exactly as p does (p included), in list order."""
        k = self.stage_key(p, stage)
        return [q for q in self.paths if stage in self._chains.get(q.pid, []) and self.stage_key(q, stage) == k]

    def owner(self, p: WorkflowView, stage: str) -> WorkflowView | None:
        """The first workflow (in list order) that computes this stage the same way, or None."""
        s = self.sharing(p, stage)
        return s[0] if s else None

    def is_pending(self, p: WorkflowView | None, stage: str) -> bool:
        return self.state(p, stage) != "done"

    def pending(self, p: WorkflowView) -> list[str]:
        return [s for s in self.all_stages(p) if self.state(p, s) != "done"]

    def deps(self, p: WorkflowView, stage: str) -> list[str]:
        if not self.ready or stage not in self._chains.get(p.pid, []):
            return []
        return list(api.deps(self.p, p.pid, stage))

    def upstream(self, p: WorkflowView, stage: str) -> set[str]:
        out, todo = set(), list(self.deps(p, stage))
        while todo:
            s = todo.pop()
            if s not in out:
                out.add(s)
                todo += self.deps(p, s)
        return out

    def downstream(self, p: WorkflowView, stage: str) -> list[str]:
        """The stages of the workflow that follow (depend on) this one, in order, the stage included."""
        return [s for s in self.all_stages(p) if s == stage or stage in self.upstream(p, s)]

    def pending_upto(self, p: WorkflowView, stages) -> list[tuple[WorkflowView, str]]:
        """What 'Run to here' computes: the given stages and everything they need, when not done yet."""
        want = set()
        for s in stages:
            want |= {s} | self.upstream(p, s)
        return [(p, x) for x in self.all_stages(p) if x in want and self.state(p, x) != "done"]

    def all_pending(self) -> list[tuple[WorkflowView, str]]:
        """What 'Run all' computes: each distinct result once, with the workflow that runs it."""
        out = []
        for j in self.plan_all().jobs:
            v = self.view(j.run_wf)
            if v is not None:
                out.append((v, j.stage))
        return out

    # ------------------------------------------------------------ durations and costs
    def seconds(self, stage: str, p: WorkflowView | None = None) -> float:
        """What a stage takes on this machine (the engine's estimate; ≈ for the methods it does not have yet)."""
        if stage in EXTRA_SECONDS:
            return EXTRA_SECONDS[stage]
        p = p or self.path
        if p is None or not self.ready:
            return 0.0
        k = ("sec", stage, self.stage_key(p, stage), self.device, self._epoch)
        got = self._est.get(k)
        if got is None:
            try:
                got = float(api.estimate(self.p, p.pid, stage, self.device).seconds)
            except (ValueError, KeyError, OSError):
                got = 0.0
            self._est[k] = got
        return got

    def path_seconds(self, p: WorkflowView) -> float:
        return sum(self.seconds(s, p) for s in self.all_stages(p))

    def _what_if(self, opts: dict, params: dict | None = None, end: str | None = None) -> Workflow:
        return Workflow(-1, "", wfl.check_options(to_engine(opts)), copy.deepcopy(params or {}),
                        "stacking" if end else None)

    def opts_seconds(self, opts: dict, params: dict | None = None) -> float:
        """A workflow with these options, from scratch (nothing reused): the sum of its stages' durations."""
        return self._opts_estimate(opts, params)[0]

    def opts_measured(self, opts: dict, params: dict | None = None) -> bool:
        """Whether every duration of ``opts_seconds`` was measured on this machine (else it is shown with ≈)."""
        return self._opts_estimate(opts, params)[1]

    def _opts_estimate(self, opts: dict, params: dict | None = None) -> tuple[float, bool]:
        if not self.ready:
            return 0.0, False
        k = ("opts", repr(sorted(opts.items())), repr(sorted((params or {}).items())), self.device, self._epoch,
             self._sources_sig())
        got = self._est.get(k)
        if got is None:
            w = self._what_if(opts, params)
            try:
                est = [api.estimate(self.p, w, s, self.device) for s in api.chain(self.p, w)]
                got = (sum(float(e.seconds) for e in est), all(e.measured for e in est))
            except (ValueError, KeyError, OSError):
                got = (0.0, False)
            self._est[k] = got
        return got

    def _sources_sig(self) -> tuple:
        """What the durations of a workflow from scratch depend on besides its options: the frames."""
        return (tuple(sorted(tuple(x) for x in self.p.excluded)), tuple(len(s.lights) for s in self.p.sessions))

    #: the device an estimate is for: the engine's names ("cuda:0" says a GPU computes, "cpu" that none does)
    DEVICE_OF = {"gpu": "cuda:0", "cpu": "cpu"}

    def _method_key(self, p: WorkflowView, block: str, method: str, device: str | None):
        """(the cache key, the options the method needs) of a method's duration, or None when the engine refuses them.
        ``device``: "gpu", "cpu", or None for the machine's."""
        tool = {"fond": {"auto": "auto", "rbf": "rbf", "simple": "simple"}, "etoiles": {"neural": "etoiles"},
                "debruitage": {"generique": "debruitage", "affine": "debruitage"},
                "deconv": {"psf": "deconv", "neural": "deconv"}}.get(block, {}).get(method)
        try:
            opts = self.with_tool(p.opts, tool)[0] if tool else dict(p.opts)      # the options the method needs
        except wfl.OptionRefused:
            return None
        if block == "debruitage":
            opts["dn"] = method
        dev = self.DEVICE_OF[device or ("gpu" if self.gpu else "cpu")]      # the machine's own: the class it computes with
        return ("method", block, repr(sorted(opts.items())), repr(sorted(p.params.items())), dev, self._epoch,
                self._sources_sig()), opts

    def method_seconds(self, p: WorkflowView, block: str, method: str,
                       device: str | None = None) -> tuple[float, bool]:
        """(seconds, measured) of a block of the views computed with this method: the stages of the block in the
        workflow that has the method. ``device``: "gpu" or "cpu", else the machine's. The methods the engine does not
        have yet (◇) are the spec's estimates. Reads manifests and the history: the worker's call, the views use
        ``method_times``."""
        from .blocks import BY_KEY
        m = next((x for x in methods_of(block) if x["key"] == method), None)
        if m is not None and m["stage"] in EXTRA_SECONDS:
            return EXTRA_SECONDS[m["stage"]], False
        if not self.ready:
            return 0.0, False
        km = self._method_key(p, block, method, device)
        if km is None:
            return 0.0, False
        k, opts = km
        got = self._est.get(k)
        if got is None:
            # Synthetic stars is a block of the graph, not an option: its stage in this workflow as it is
            w = p.pid if block == "synth" else self._what_if(opts, p.params)
            try:
                chain = api.chain(self.p, w)
                dev = k[4]
                est = [api.estimate(self.p, w, s, dev) for s in BY_KEY[block]["stages"] if s in chain]
                got = (sum(e.seconds for e in est), all(e.measured for e in est))
            except (ValueError, KeyError, OSError):
                got = (0.0, False)
            self._est[k] = got
        return got

    def method_times(self, p: WorkflowView, block: str, method: str) -> dict[str, tuple[float, bool]] | None:
        """What a method takes on each device it runs on, for the card: {"gpu": (seconds, measured), "cpu": ...} (a
        method that runs on the CPU only has just that one). The GUI thread only reads what the worker has already
        estimated: None while it has not (the worker is asked, ``costChanged`` follows when the durations are in)."""
        m = next((x for x in methods_of(block) if x["key"] == method), None)
        if m is None:
            return {}
        devs = ("cpu",) if m["runs"] == "cpu" else ("gpu", "cpu")
        if m["stage"] in EXTRA_SECONDS:                        # the spec's own figure, for the device it is meant for
            return {devs[0]: (EXTRA_SECONDS[m["stage"]], False)}
        if not self.ready:
            return None
        out = {}
        for dev in devs:
            km = self._method_key(p, block, method, dev)
            if km is None:
                out[dev] = (0.0, False)
                continue
            got = self._est.get(km[0])
            if got is None:
                if not self._est_wanted:                       # its own channel: never replaces the first prefetch
                    self._est_wanted = True
                    self.worker.submit(f"model.wanted:{id(self)}", self._prefetch_estimates, False)
                return None
            out[dev] = got
        return out

    def cost_of(self, opts: dict, params: dict | None = None) -> tuple[float, list[str], list[str]]:
        """Cost of a workflow with these options: (seconds to compute, stages to compute, stages reused)."""
        if not self.ready:
            return 0.0, [], []
        w = self._what_if(opts, params)
        plan = api.plan_workflows(self.p, [w], self.device, self._history_est)
        todo = [j.stage for j in plan.jobs]
        return float(plan.seconds), todo, [s for s in api.chain(self.p, w) if s not in todo]

    def cost(self, workflows=None) -> api.Cost:
        """What running these workflows (ids or what-if Workflows; None: all) would take: the plan's totals."""
        if not self.ready:
            return api.Cost(0.0, True, 0.0, 0)
        k = None
        if workflows is None or not any(isinstance(w, Workflow) for w in workflows):
            k = ("cost", None if workflows is None else tuple(workflows), self.device)
            if k in self._cache:
                return self._cache[k]
        try:
            c = api.cost(self.p, workflows, self.device, self._history_est)
        except (ValueError, KeyError, OSError):
            c = api.Cost(0.0, True, 0.0, 0)
        if k is not None:
            self._cache[k] = c
        return c

    def total_seconds(self) -> float:
        """The time of Run all (the Run button)."""
        return self.cost().seconds

    def plan_all(self):
        """The plan of Run all (api.Plan): the distinct jobs not done yet."""
        return api.plan_all(self.p, self.device, self._history_est) if self.ready else api.Plan()

    def plan_to_block(self, p: WorkflowView, key: str):
        """The plan of Run to here for a block of the views (kept until the next sync: planning reads the manifests
        of the store)."""
        if not self.ready:
            return api.Plan()
        k = ("plan_to", p.pid, key, self.device)
        if k not in self._cache:
            self._cache[k] = api.plan_to_block(self.p, p.pid, engine_name(key), self.device, self._history_est)
        return self._cache[k]

    # ------------------------------------------------------------ breakpoints: pause for the user to check
    # A breakpoint is a block of one workflow ("Review pause"): a run stops after it, and goes on only when
    # the user says so (Continue). It lives in the definition (``Workflow.breakpoints``, engine names) and is no part of
    # any key. What is reviewed is tied to the block's results: a fingerprint of its stage keys, kept in state.json.
    def has_breakpoint(self, p: WorkflowView, key: str) -> bool:
        """Whether this workflow pauses after this block (a block of the views)."""
        return engine_name(self._bp_key(p, key)) in self.p.workflow(p.pid).breakpoints     # a repeated block too (1.2)

    def _bp_key(self, p: WorkflowView, key: str) -> str:
        """The block a pause is set on: a session's Calibration drawn beside its session ("calibration@2") is the
        workflow's own Calibration when the workflow stacks that one session (the engine computes it so)."""
        if "@" in key and self.ready:
            keys = self.pause_keys(p)
            base = key.partition("@")[0]
            if key not in keys and base in keys:
                return base
        return key

    def breakpoints(self, p: WorkflowView) -> list[str]:
        """The blocks (of the views) this workflow pauses after, in the graph's order; only those it computes (a
        block switched off keeps its pause, which applies again once it is on). A session's Calibration
        ("calibration@2") is one of them: it comes first."""
        have = {view_key(b) for b in self.p.workflow(p.pid).breakpoints if wfl.base(b) in ENGINE_KEY}
        return [k for k in self.pause_keys(p) if k in have]

    def waiting_breakpoints(self, paths=None) -> list[tuple[int, str]]:
        """The pauses kept on blocks switched off, (pid, key of the views): a run goes through them until the block is
        on again (GUI exploration round 3: the run window must not announce them, nor stay silent)."""
        if not self.ready:
            return []
        out = []
        for q in (self.paths if paths is None else paths):
            have = [view_key(b) for b in self.p.workflow(q.pid).breakpoints if wfl.base(b) in ENGINE_KEY]
            out += [(q.pid, k) for k in have if self.is_cut(q, k)]       # kept in the graph's order (set_breakpoint)
        return out

    def pause_keys(self, p: WorkflowView) -> list[str]:
        """The blocks of a workflow a pause can follow, in the order they run: each session's Calibration, then the
        chain (``block_keys``)."""
        return [view_key(b.block) for b in self._blocks.get(p.pid, []) if wfl.base(b.block) in ENGINE_KEY]

    def pause_name(self, pid: int, key: str) -> str:
        """A paused block as the graph names it: "Gradient 1" when the workflow has two, a session's Calibration with
        its session ("Calibration · Session 2")."""
        from .blocks import block_title
        q = self.view(pid)
        base, at, sid = key.partition("@")
        # a workflow's own Calibration drawn beside its one session: named with it
        one = q is not None and q.source is None and len(self.sessions_of(q)) == 1
        if base == "calibration" and not at and one and self.separate_sessions():
            at, sid = "@", str(self.sessions_of(q)[0])
        text = _("Output") if key == "post" else block_title(q, base) if q is not None else block_title(None, base)
        if at:
            try:
                text += " · " + (self.p.session(int(sid)).name or _("Session {n}").format(n=sid))
            except (KeyError, ValueError):
                pass
        return text

    def set_breakpoint(self, paths, key: str, on: bool = True):
        """Pause after this block in these workflows (or no longer). One change for Undo; saved with the project."""
        eb = engine_name(key)                             # a repeated block: its own node ("gradient~2", 1.2)
        if wfl.base(eb) not in wfl.BLOCKS:
            raise ValueError(f"no breakpoint on {key!r}: it is not a block of the engine")
        with self._edit():
            for q in paths:
                eb = engine_name(self._bp_key(q, key))
                w = self.p.workflow(q.pid)
                have = set(w.breakpoints)
                w.breakpoints = sorted(have | {eb} if on else have - {eb}, key=wfl.block_order)

    def _prune_post(self):
        """Post follows the graph (GUI exploration 3.3): a layer of a Split image goes with its Split block (the
        workflow's own image then comes back to Post), a layer of a workflow gone goes."""
        ws = {w.id: w for w in self.p.workflows}
        for v in self.p.post["versions"]:
            self._prune_version(v, ws)

    @staticmethod
    def _prune_version(v: dict, ws: dict):
        layers = v.get("layers") or []
        keep, gone, changed = [], set(), False
        for x in layers:
            if x.get("kind") != "workflow":
                keep.append(x)
                continue
            w = ws.get(x.get("pid"))
            split = w is not None and (bool((w.options or {}).get("stars")) or (
                isinstance(w.source, dict) and bool((w.source.get("fusion") or {}).get("stars"))))
            if w is None or (x.get("part") is not None and not split):
                gone.add(x.get("id"))
                changed = True
                continue
            keep.append(x)
        if changed:
            v["layers"] = [x for x in keep if not (x.get("kind") == "filter" and x.get("target") in gone)]

    def _prune_breakpoints(self):
        """A block that left a workflow takes its breakpoint with it (it would otherwise come back with the block)."""
        if not self.ready:
            return
        for w in self.p.workflows:
            if not w.breakpoints:
                continue
            try:
                inside = {b.block for b in api.blocks(self.p, w.id)}
                if w.blocks is not None:                 # a block switched off keeps its pause (it comes back with it)
                    inside |= {n["name"] if n["block"] in wfl.IMAGE_OPS else n["block"] for n in wfl.graph_of(w)}
            except (ValueError, KeyError, OSError):
                continue
            if any(b not in inside for b in w.breakpoints):
                w.breakpoints = [b for b in w.breakpoints if b in inside]

    def review_key(self, p: WorkflowView, key: str) -> str:
        """What a review is tied to: the block's results, that is the keys of its stages (the engine's ``blocks``);
        for Selection, which has no stage, the Registration it follows, the thresholds and the frames excluded by hand."""
        key = self._bp_key(p, key)
        k = ("rk", p.pid, key)
        if k not in self._cache:
            def ref(vk):
                # a repeated block ("fond~2") and a session's Calibration ("calibration@2") have their own
                return next((b for b in self._blocks.get(p.pid, []) if view_key(b.block) == vk), None)
            b = ref(key)
            parts: list = [KEY_ENGINE_BLOCK.get(key, key), list(b.key) if b is not None else []]
            if key == "crop":
                reg = ref("alignement")
                parts += [list(reg.key) if reg is not None else [], (p.params.get("stack") or {}).get("zone")]
            if key == "selection":
                reg, st = ref("alignement"), p.params.get("stack") or {}
                parts += [list(reg.key) if reg is not None else [], st.get("min_transparency"),
                          st.get("max_fwhm_ratio"), st.get("selection", "min_weight"),
                          st.get("selection_transparency", "on"), sorted(list(x) for x in self.frames_excluded)]
            raw = json.dumps(parts, sort_keys=True, default=str)
            self._cache[k] = f"{KEY_ENGINE_BLOCK.get(key, key)}:{hashlib.sha1(raw.encode()).hexdigest()[:16]}"
        return self._cache[k]

    def is_reviewed(self, p: WorkflowView, key: str) -> bool:
        """Whether the user marked the block's present results as checked."""
        return self.review_key(p, key) in self._reviewed

    def review_state(self, p: WorkflowView, key: str) -> str | None:
        """None (no breakpoint here) | "pending" (the run will stop here) | "paused" (computed, waiting for the user:
        the first review of the workflow) | "reviewed" (computed, checked: the run goes through)."""
        if not self.has_breakpoint(p, key):
            return None
        if not self.ready or block_state(self, p, key) != "done":
            return "pending"
        if self.is_reviewed(p, key):
            return "reviewed"
        first = next((k for k in self.breakpoints(p) if not self.is_reviewed(p, k)), None)
        return "paused" if self._bp_key(p, key) == first else "pending"

    def pause_point(self, only: WorkflowView | None = None, upto: str | None = None) -> Pause | None:
        """The review a run stops at: the first block, in the graph's order, that has a breakpoint whose results are
        not reviewed, in any workflow (``only``: in this one, up to the block ``upto``). None when the run goes
        through."""
        if not self.ready:
            return None
        found: list[tuple[tuple, int, WorkflowView, str]] = []
        later: list[tuple[int, str]] = []
        for i, q in enumerate(self.paths if only is None else [only]):
            keys = self.pause_keys(q)
            reach = keys[:keys.index(upto) + 1] if upto in keys else keys
            todo = [k for k in self.breakpoints(q) if k in reach and not self.is_reviewed(q, k)]
            if todo:
                # the graph's order: the block's column, then a repeated block's number ("fond~2" after "fond", 1.2)
                found.append(((COLUMNS.index(kbase(todo[0])) if kbase(todo[0]) in COLUMNS else len(COLUMNS),
                               int(knum(todo[0]) or 0)), i, q, todo[0]))
                later += [(q.pid, k) for k in todo[1:]]        # each workflow's later pauses, under its own name
        if not found:
            return None
        found.sort(key=lambda t: t[:2])
        block = found[0][3]
        pids = [q.pid for _c, _j, q, k in found if k == block]
        # Run all: every workflow stops at its own first review (GUI exploration: only the one paused the most
        # upstream went on); the first in the graph's order names the pause
        stops = [(q.pid, k) for _c, _j, q, k in found]
        return Pause(block, pids, all(block_state(self, q, k) == "done" for _c, _j, q, k in found), later, stops)

    def mark_reviewed(self, pause: Pause):
        """Continue: the results the user has looked at are marked as checked. Only a block that is computed now is
        marked, with the results it has now: if the user changed something that made it Outdated meanwhile, the
        breakpoint applies again to the new results."""
        for pid, block in pause.all_stops():
            q = self.view(pid)
            if q is not None and self.ready and block_state(self, q, block) == "done":
                k = self.review_key(q, block)
                if k not in self._reviewed:
                    self._reviewed.append(k)
        self._save_reviewed()
        self._cache.clear()
        self.pathsChanged.emit()

    def _take_reviewed(self, kept):
        """The reviews read from state.json (worker): added to what this session marked."""
        new = [k for k in kept if k not in self._reviewed]
        if new:
            self._reviewed = [*new, *self._reviewed]
            self._cache.clear()

    def _save_reviewed(self):
        del self._reviewed[:-REVIEWED_KEPT]
        if self.readonly or self.root is None:
            return
        try:
            state = read_state(self.root)
            state["reviewed"] = list(self._reviewed)
            write_state(self.root, state)
        except OSError:
            pass                                             # the review is kept for this session

    def plan_run(self) -> tuple[api.Plan, Pause | None]:
        """Run all, with the breakpoints: each workflow up to its own first review not yet done, the others whole,
        else everything. Returns the plan and the pause it ends at (None: the whole project)."""
        pause = self.pause_point()
        return (self.plan_all(), None) if pause is None else (self._plan_pause(pause, everyone=True), pause)

    def plan_run_to(self, p: WorkflowView, key: str) -> tuple[api.Plan, Pause | None]:
        """Run to here, with the breakpoints: the block, or the first review of this workflow before it."""
        pause = self.pause_point(only=p, upto=key)
        return (self.plan_to_block(p, key), None) if pause is None else (self._plan_pause(pause), pause)

    def _plan_pause(self, pause: Pause, everyone: bool = False) -> api.Plan:
        """The plans to each workflow's own review (``pause.all_stops``), as one plan: a job two plans share is kept
        once (the jobs of a plan come after those they read, so the concatenation stays in order). ``everyone`` (Run
        all): the workflows without a pause are computed whole too, unless that computes past a paused block (a
        fusion of a paused channel): those wait for the review (``pause.waiting``)."""
        views = {q.pid: q for q in self.paths}
        plans = [self.plan_to_block(views[pid], k) for pid, k in pause.all_stops() if pid in views]
        if everyone:
            stopped = {pid for pid, _k in pause.all_stops()}
            past: set = set()                                # the jobs after a paused block: not before the review
            upto = {j.key for pl in plans for j in pl.jobs}
            for pid in stopped & set(views):
                past |={j.key for j in api.plan_workflows(self.p, [pid], self.device, self._history_est).jobs} - upto
            for q in self.paths:
                if q.pid in stopped:
                    continue
                pl = api.plan_workflows(self.p, [q.pid], self.device, self._history_est)
                if past & {j.key for j in pl.jobs}:
                    pause.waiting.append(q.pid)
                else:
                    plans.append(pl)
        return self._one_plan(plans)

    def plan_align(self, p: WorkflowView) -> api.Plan:
        """Run to here on Align images: every image of the block; one going straight to Post, its Output too (Post shows
        it aligned at once)."""
        return self._one_plan([self.plan_to_block(q, "post" if self.is_plain_image(q) and "render_simple" in self.all_stages(q)
                                                  else "image") for q in self.align_group(p)])

    @staticmethod
    def _one_plan(plans: list) -> api.Plan:
        """Several plans as one: a job two share is kept once (the jobs of a plan come after those they read)."""
        jobs: dict = {}
        for pl in plans:
            for j in pl.jobs:
                jobs.setdefault(j.key, j)
        out = api.Plan(list(jobs.values()), [t for pl in plans for t in pl.targets], {},
                       plans[0].rules if plans else None)
        for pl in plans:
            out.skipped.update(pl.skipped)
        return out

    # ------------------------------------------------------------ the toolbox: options and their rules
    @staticmethod
    def has_tool(o: dict, key: str) -> bool:
        return {"debruitage": bool(o.get("debruitage")), "etoiles": o.get("etoiles") == "retirer",
                "auto": o.get("fond") == "auto", "rbf": o.get("fond") == "rbf",
                "simple": o.get("fond") == "simple", "deconv": bool(o.get("deconv"))}[key]

    @staticmethod
    def with_tool(o: dict, key: str) -> tuple[dict, list[str]]:
        """The options with this tool added, and what came with it (the notes are English texts to translate)."""
        opt, add, _rm = TOOL_OPTION[key]
        new, notes = wfl.set_option(to_engine(o), opt, add)
        return {**o, **from_engine(new)}, list(notes)

    @staticmethod
    def without_tool(o: dict, key: str) -> tuple[dict | None, list[str]]:
        """The options with this tool removed and what went with it, or (None, [reason]) when it cannot go."""
        opt, _add, rm = TOOL_OPTION[key]
        try:
            new, notes = wfl.set_option(to_engine(o), opt, rm)
        except wfl.OptionRefused as e:
            return None, [str(e)]
        return {**o, **from_engine(new)}, list(notes)

    def tool_cost(self, o: dict, key: str) -> float:
        """What adding this tool to a workflow costs: the difference of the two workflows' durations."""
        new, _notes = self.with_tool(o, key)
        return self.opts_seconds(new) - self.opts_seconds(o)

    def set_opts(self, p: WorkflowView, opts: dict):
        """The options of a workflow, replaced as a whole (the rules were applied by with_tool / without_tool)."""
        with self._edit():
            ws = api.workflows(self.p)
            w = next(x for x in ws if x.id == p.pid)
            w.options = wfl.check_options(to_engine(opts))
            api.set_workflows(self.p, ws, self.p.current)

    # ------------------------------------------------------------ workflows
    def unique_name(self, base: str, numbered: bool = False) -> str:
        """``base``, else ``base 2``, ``base 3``... ``numbered``: one of several made together, numbered from 1 as a
        block placed twice ("Image 1", "Image 2", issue #134); a series already numbered goes on numbered, so that the
        workflow's name says the number its block shows."""
        names = {p.name for p in self.paths}
        series = any(x.startswith(base + " ") and x[len(base) + 1:].isdigit() for x in names)
        if base not in names and not numbered and not series:
            return base
        n = 1 if numbered or base not in names else 2
        while f"{base} {n}" in names:
            n += 1
        return f"{base} {n}"

    def add_path(self, name: str, opts: dict, params: dict | None = None, end: str | None = None) -> WorkflowView:
        """A new workflow. Without settings of its own it joins the current one's pre-processing (the settings of
        Calibration to Stacking are copied), so that the two share those blocks. ``end``: "empilement", its output
        leaves from Stacking (a raw stack)."""
        params = copy.deepcopy(params or {})
        if not params and self.path is not None:
            params = {s: dict(v) for s, v in self.path.params.items() if s in TRUNK}
        # the camera white balance in Stacking (issue #6): on in a new project; else as the project's workflows have
        # it, so that they share their stack (a project computed before it stays without)
        have = [w.params.get("stack", {}).get("white_balance", False) for w in self.p.workflows]
        if (bool(have[0]) if have else True):
            params.setdefault("stack", {})["white_balance"] = True
        with self._edit():
            w = api.add_workflow(self.p, name, wfl.check_options(to_engine(opts)), params)
            if end:
                self.p.workflow(w.id).end = wfl.check_end(KEY_ENGINE_BLOCK.get(end, end))
        return self._view_of(w.id)

    def _view_of(self, wid: int) -> WorkflowView:
        """The view of a workflow just made (inside a batch the views are built at its end: built here)."""
        if self.view(wid) is None:
            self._rebuild_views()
        return self.view(wid)

    def batch(self):
        """Several edits that Undo takes back as one: ``with model.batch(): ...``."""
        return self._edit()

    def duplicate(self, src: WorkflowView, name: str) -> WorkflowView:
        """A copy of a workflow: its options, settings and exit."""
        with self._edit():
            w = api.add_workflow(self.p, self.unique_name(name), None, None, copy_of=src.pid)
            x, s0 = self.p.workflow(w.id), self.p.workflow(src.pid)
            x.breakpoints = list(s0.breakpoints)
            x.source, x.sessions = copy.deepcopy(s0.source), copy.deepcopy(s0.sessions)
            if s0.blocks is not None:                     # a graph stays a graph, with its own blocks (3.2)
                self.p.set_blocks(w.id, copy.deepcopy(s0.blocks))
            self.p.validate()
        return self.view(w.id)

    def rename(self, pid: int, name: str) -> bool:
        """A workflow's new name; False when it does not change or is refused (``name_problem`` says why)."""
        name = name.strip()
        if name == self.p.workflow(pid).name or self.name_problem(name, pid=pid):
            return False
        with self._edit():
            self._rename(pid, name)
        return True

    def _rename(self, pid: int, name: str):
        old = self.p.workflow(pid).name
        api.rename_workflow(self.p, pid, name)
        for v in self.p.post["versions"]:                 # its layers named after it follow (GUI exploration 3.4)
            v["layers"] = [
                {**x, "name": self.p.workflow(pid).name + x["name"][len(old):]}
                if x.get("kind") == "workflow" and x.get("pid") == pid and str(x.get("name", "")).startswith(old)
                else x for x in v.get("layers") or []]

    @staticmethod
    def branch_name(flow: str, key: str) -> str:
        """The name of a block's image sent to Post: its workflow's and its block's ("Ha · Denoise")."""
        from .blocks import block_title
        return f"{flow} · " + (_("Raw stack") if key == "empilement" else block_title(None, key.partition("~")[0]))

    def _follow_branch_names(self):
        """A branch sent to Post keeps a name that says what it is: blocks added or taken after its last one, its
        name follows ("Ha · Gradient" becomes "Ha · Denoise"; GUI exploration 2). A name the user gave stays."""
        from .blocks import BY_KEY
        titles = {self.branch_name("", k)[3:] for k in BY_KEY}
        names = {w.name for w in self.p.workflows}
        for w in self.p.workflows:
            flow, sep, title = w.name.partition(" · ")
            if not sep or w.blocks is None or flow not in names - {w.name}:
                continue
            base = title.rsplit(" ", 1)[0] if title not in titles and title.rsplit(" ", 1)[-1].isdigit() else title
            on = [n for n in w.blocks if not n.get("off") and n.get("block") != "output"]
            if base not in titles or not on:
                continue
            want = self.branch_name(flow, node_key(on[-1]))
            if want != f"{flow} · {base}":
                self._rename(w.id, self.unique_name(want))

    def delete_workflow(self, p: WorkflowView) -> list[str]:
        """Delete workflow (GUI exploration 2): a fusion takes back what it added (``remove_fusion``); a flow of frames
        goes with the Sessions only it read (a session no workflow reads would stay unwired, its frames said used),
        the project's last ones excepted. Returns the names of the sessions gone (their files are never touched)."""
        if is_fusion_view(p):
            self.remove_fusion(p)
            return []
        w = self.p.workflow(p.pid)
        mine = [s.id for s in self.p.sessions_of(w.id)] if w.source is None and w.sessions is not None else []
        others = [x for x in self.p.workflows if x.id != w.id and x.source is None]
        alone = [sid for sid in mine if not any(sid in [s.id for s in self.p.sessions_of(x.id)] for x in others)]
        if len(alone) >= len(self.p.sessions):
            alone = []                                    # the project keeps its sessions
        names = [self.p.session(sid).name for sid in alone]
        with self._edit():
            self.remove_path(self.paths.index(p))
            for sid in alone:
                self.remove_session(sid)
        return names

    def remove_path(self, i: int):
        pid = self.paths[i].pid
        with self._edit():
            api.delete_workflow(self.p, pid)
        self.current = min(self.current, len(self.paths) - 1)
        self.currentChanged.emit()

    def add_exit(self, src: WorkflowView, end: str, name: str) -> WorkflowView:
        """A new workflow: the same blocks as src up to ``end`` (Stacking), whose output goes straight to Post."""
        with self._edit():
            w = api.add_exit(self.p, src.pid, KEY_ENGINE_BLOCK.get(end, end), self.unique_name(name))
        return self.view(w.id)

    def scope_paths(self) -> list[WorkflowView]:
        if self.scope:
            out = [q for q in self.paths if q.pid in self.scope]
            if out:
                return out
        return [self.path] if self.path is not None else []

    # ------------------------------------------------------------ changes and variants
    @staticmethod
    def _api_changes(ch: dict) -> list[dict]:
        """A change of the views as the engine's changes (``api.Change``); ◇ raises NotInEngine."""
        if ch["kind"] == "params":
            return [{"kind": "param", "stage": st, "name": nm, "value": v} for st, nm, v in ch["items"]]
        if ch["kind"] == "param":
            v = ch.get("value")
            if v is not None and ch.get("suggested") is not None and v == ch["suggested"]:
                v = None
            return [{"kind": "param", "stage": ch["stage"], "name": ch["name"], "value": v}]
        key, v = ch["key"], ch["value"]
        if key == "dn":
            if v not in DN_ENGINE:
                raise NotInEngine(NOT_IN_ENGINE)
            return [{"kind": "opt", "key": "denoise", "value": DN_ENGINE[v]}]
        if key == "fond":
            return [{"kind": "opt", "key": "gradient", "value": GRADIENT_ENGINE.get(v, "polynomial")}]
        if key == "debruitage":
            return [{"kind": "opt", "key": "denoise", "value": "pretrained" if v else None}]
        if key == "etoiles":
            return [{"kind": "opt", "key": "stars", "value": "neural" if v == "retirer" else None}]
        raise NotInEngine(NOT_IN_ENGINE)

    def _changed(self, p: WorkflowView, ch: dict) -> Workflow:
        """What-if: a copy of the workflow with the change made. Each step is copied again: the API keeps its keys
        by object, and ``with_change`` has already asked the keys of the copy before changing it (the suggestion of
        an automatic setting needs them), so that the copy it returns still answers the keys it had."""
        w = p.pid
        for c in self._api_changes(ch):
            w = copy.deepcopy(api.with_change(self.p, w, c))
        return w

    def throws_away(self, paths, ch) -> bool:
        """Whether the change would make something already computed in one of these workflows Outdated. (A stage
        name means: something computed at or after that stage.)"""
        if not self.ready:
            return False
        if isinstance(ch, str):
            return any(self.state(q, x) != "pending" for q in paths for x in self.downstream(q, ch))
        try:
            for q in paths:
                c = self._changed(q, ch)
                now = set(api.chain(self.p, c))
                for stage in self.all_stages(q):
                    if self.state(q, stage) == "done" and (
                            stage not in now or api.stage_key(self.p, c, stage) != self.stage_key(q, stage)):
                        return True
        except NotInEngine:
            return False
        return False

    def cost_with(self, paths, ch) -> float:
        """What would have to be computed, in seconds, once the change is made in these workflows."""
        return self.cost_of_change(paths, ch).seconds

    def cost_of_change(self, paths, ch) -> api.Cost:
        """The plan's totals (seconds, measured or ≈, disk) once the change is made in these workflows."""
        if not self.ready:
            return api.Cost(0.0, True, 0.0, 0)
        try:
            return self.cost([self._changed(q, ch) for q in paths])
        except NotInEngine:
            return api.Cost(0.0, True, 0.0, 0)

    def apply_change(self, paths, ch) -> list[str]:
        """Replace: the change goes to these workflows; returns the notes (English texts to translate)."""
        notes: list[str] = []
        with self._edit():
            for c in self._api_changes(ch):
                notes += [n for n in api.apply_change(self.p, [q.pid for q in paths], c) if n not in notes]
        for q in paths:
            if ch["kind"] == "param":
                self._patch_rows([q.pid], ch["stage"], ch["name"], ch.get("value"))
        return notes

    def add_variant(self, p: WorkflowView, ch, title: str) -> WorkflowView:
        """A copy of p with the change; p stays as it was. Both share everything before the changed block."""
        cs = self._api_changes(ch)
        with self._edit():
            w = api.make_variant(self.p, p.pid, cs[0], self.unique_name(title))
            x, s0 = self.p.workflow(w.id), self.p.workflow(p.pid)
            x.breakpoints = list(s0.breakpoints)
            # its source and sessions, and a graph stays a graph: the original's blocks, the change's options on them
            # (control round: made from options alone, it had the template's chain, a Selection block "Keep" added)
            x.source, x.sessions = copy.deepcopy(s0.source), copy.deepcopy(s0.sessions)
            if s0.blocks is not None:
                o = wfl.check_options(x.options)
                self.p.set_blocks(w.id, wfl.sync_blocks(copy.deepcopy(s0.blocks), wfl.source_of(x), o))
            for c in cs[1:]:
                api.apply_change(self.p, [w.id], c)
        return self.view(w.id)

    def keep_variant(self, q: WorkflowView):
        """The variant replaces its original, and takes its name."""
        with self._edit():
            api.keep_variant(self.p, q.pid)
        self.current = self.index_of(q.pid)
        self.currentChanged.emit()

    def drop_variant(self, q: WorkflowView):
        with self._edit():
            api.discard_variant(self.p, q.pid)
        v = self.view(self.p.current) if self.p.current is not None else None
        self.current = self.paths.index(v) if v in self.paths else min(self.current, len(self.paths) - 1)
        self.currentChanged.emit()

    # ------------------------------------------------------------ the settings of a stage
    def params(self, p: WorkflowView, stage: str, request: bool = True) -> list | None:
        """The settings of a stage of a workflow (api.ParamInfo list, from ``describe``), or None while the worker
        has not answered. After a change to the stage the last rows are kept, with the new value patched in, until
        the worker's answer replaces them."""
        if not self.ready or stage not in self._chains.get(p.pid, []):
            return None
        key = self.stage_key(p, stage)
        e = self._pc.get((p.pid, stage))
        fresh = e is not None and e["key"] == key and e["epoch"] == self._epoch
        if not fresh and request and not (e is not None and e.get("asked") == (key, self._epoch)):
            e = self._pc.setdefault((p.pid, stage), {"key": None, "epoch": -1, "rows": None, "asked": None})
            e["asked"] = (key, self._epoch)
            self.worker.submit(f"model.describe:{id(self)}:{p.pid}:{stage}", _describe_job, self.p, p.pid, stage)
        e = self._pc.get((p.pid, stage))
        return e["rows"] if e is not None else None

    def params_error(self, p: WorkflowView, stage: str) -> str:
        """Why the settings of a stage cannot be read (the worker failed three times), or ''."""
        e = self._pc.get((p.pid, stage))
        return (e or {}).get("error", "")

    def param_info(self, p: WorkflowView, stage: str, name: str):
        rows = self.params(p, stage)
        return next((r for r in rows if r.name == name), None) if rows else None

    def _got_params(self, wid: int, stage: str, key: str, rows: list):
        v = self.view(wid)
        if v is None:
            return
        for r in rows:
            if r.reason and r.origin == "auto":
                self._reasons[(stage, r.name)] = (r.reason, r.reason_args)
        old = self._pc.get((wid, stage))
        cur = self.stage_key(v, stage) if stage in self._chains.get(wid, []) else None
        self._pc[(wid, stage)] = {"key": key, "epoch": self._epoch, "rows": rows, "asked": None, "tries": 0}
        if cur is not None and cur != key:
            self.params(v, stage)                               # an edit came meanwhile: ask again
        if old is None or _sig(old.get("rows")) != _sig(rows):
            self.paramsChanged.emit(wid, stage)
            if stage in PREFETCH:
                self.costChanged.emit()

    def reason_for(self, stage: str, name: str):
        """The last reason an automatic value had (template, args): shown for the suggestion of a setting the user
        has since set."""
        return self._reasons.get((stage, name))

    def param_value(self, p: WorkflowView, stage: str, name: str):
        """The effective value of a setting: the stored one, else the one the engine gave (settings of the stage in
        the cache), else the static default."""
        st = p.params.get(stage) or {}
        if name in st:
            return st[name]
        e = self._pc.get((p.pid, stage))
        info = next((r for r in (e["rows"] or []) if r.name == name), None) if e else None
        return info.value if info is not None else self.default_of(stage, name)

    def default_of(self, stage: str, name: str):
        if not self.ready:                                  # the stage graph is imported by the worker (warm)
            return None
        try:
            return wfl.chains().stages[stage].Params.model_fields[name].default
        except (KeyError, AttributeError):
            return None

    def set_param(self, p: WorkflowView | None, stage: str, name: str, value, suggested=None):
        """Store a setting in the scope workflows (or in p). ``value`` None, or equal to the suggestion, goes back
        to the suggestion. Raises ValueError naming the setting when the value is not valid."""
        ids = [q.pid for q in (self.scope_paths() if p is None else [p])]
        if value is not None and suggested is not None and value == suggested:
            value = None
        with self._edit():
            api.set_param(self.p, ids, stage, name, value)
        self._patch_rows(ids, stage, name, value)

    def _patch_rows(self, ids, stage, name, value):
        """The cached rows show the new value at once; the worker's answer will confirm it."""
        for wid in ids:
            e = self._pc.get((wid, stage))
            if not e or not e.get("rows"):
                continue
            rows = []
            for r in e["rows"]:
                if r.name == name:
                    r = copy.copy(r)
                    r.value = r.suggested if value is None else value
                    r.origin = "default" if value is None else "user"
                rows.append(r)
            e["rows"] = rows

    def _unpatch_rows(self):
        """Undo / Redo: the cached rows patched by a change (``_patch_rows``) show the settings put back (a setting
        undone showed its undone value until the worker answered: Post's Linked channels stayed unticked)."""
        for (wid, stage), e in list(self._pc.items()):
            v = self.view(wid)
            if v is None or not e or not e.get("rows"):
                continue
            st = v.params.get(stage) or {}
            for r in e["rows"]:
                if r.name in st and (r.value != st[r.name] or r.origin != "user"):
                    self._patch_rows([wid], stage, r.name, st[r.name])
                elif r.name not in st and r.origin == "user":
                    self._patch_rows([wid], stage, r.name, None)

    def set_trunk_param(self, stage: str, name: str, value) -> bool:
        """A decision taken in the sources: it counts for every workflow. False when the engine has no such setting
        (◇)."""
        try:
            with self._edit():
                api.set_param(self.p, [q.pid for q in self.paths], stage, name, value)
        except ValueError:
            return False
        return True

    # ------------------------------------------------------------ frame selection (Selection and Stacking)
    def fwhm_median(self) -> float | None:
        """The median star size of the registered frames, when Registration is done."""
        v = self._facts.register.get("fwhm_px")
        return float(v) if v else None

    def selection_rule(self, p=None) -> str:
        """The frame selection rule of the workflow: "min_weight" (the default), "sota" (k × σ filters) or "thresholds" (the
        transparency and star-size thresholds)."""
        p = p or self.path
        v = self.param_value(p, "stack", "selection") if p is not None else None
        return str(v) if v else "min_weight"

    def thresholds_active(self, p=None) -> bool:
        """Whether the transparency and star-size thresholds act: only with the "thresholds" rule."""
        return self.selection_rule(p) == "thresholds"

    def thresholds_unit(self) -> str:
        return "px" if self.fwhm_median() else "× median"

    def thresholds_now(self) -> tuple[float, float]:
        """(minimum transparency in % of the best frames, maximum star size in px, or × the median while unknown)."""
        p = self.path
        if p is None:
            return 60.0, 2.0
        t = float(self.param_value(p, "stack", "min_transparency") or 0.0)
        r = float(self.param_value(p, "stack", "max_fwhm_ratio") or 2.0)
        m = self.fwhm_median()
        return 100 * t, r * m if m else r

    def suggested_thresholds(self) -> tuple[float, float]:
        p = self.path
        it = self.param_info(p, "stack", "min_transparency") if p is not None else None
        ir = self.param_info(p, "stack", "max_fwhm_ratio") if p is not None else None
        t = it.suggested if it is not None else self.default_of("stack", "min_transparency")
        r = ir.suggested if ir is not None else self.default_of("stack", "max_fwhm_ratio")
        return 100 * float(t or 0.0), float(r or 2.0) * (self.fwhm_median() or 1.0)

    def set_selection(self, tmin: float | None = None, fmax: float | None = None, hand=None):
        """The Selection thresholds (for the scope workflows) and the frames excluded by hand (for the project:
        (session id, file name) pairs)."""
        if hand is not None:
            cur = self.frames_excluded
            self.set_excluded(set(hand) - cur, True)
            self.set_excluded(cur - set(hand), False)
        t0, f0 = self.thresholds_now()
        tmin = t0 if tmin is None else tmin
        fmax = f0 if fmax is None else fmax
        if abs(tmin - t0) > 1e-9 or abs(fmax - f0) > 1e-9:
            m = self.fwhm_median()
            self.apply_change(self.scope_paths(), {"kind": "params", "items": [
                ("stack", "min_transparency", round(tmin / 100, 4)),
                ("stack", "max_fwhm_ratio", round(fmax / m if m else fmax, 4))]})
        self.costChanged.emit()

    def set_excluded(self, pairs, excluded: bool = True):
        """Exclude (or keep again) light frames by hand, for the whole project: (session id, file name) pairs."""
        pairs = list(pairs)
        if not pairs:
            return
        with self._edit():
            api.set_excluded(self.p, pairs, excluded)
        self._sel, self._sel_want = None, None
        self.sourcesChanged.emit()

    @staticmethod
    def _note_left_out(s, kind: str, paths: list[str], left: bool):
        """A session's record of its calibration frames left out by hand (they are shown, to be taken back)."""
        norm = {norm_path(x) for x in paths}
        kept = [f for f in s.left_out.get(kind, []) if norm_path(f) not in norm]
        if left:
            kept += [Path(x).as_posix() for x in paths]
        if kept or kind in s.left_out:
            s.left_out[kind] = kept

    def set_calibration_excluded(self, session: int, kind: str, path: str, excluded: bool,
                                 folder: list[tuple[str, bool]] | None = None) -> bool:
        """Leave a calibration frame out of its master, or take it back (``kind``: darks, flats, bias or flat_darks):
        it leaves the session's list and is noted, to be shown and taken back; a folder a project made before lists
        becomes its frames. ``folder``: the frames of that folder and whether each is left out now, as the worker
        listed them (``services/inspect.browse_group``); None: the session's own lists say it. The last frame can go
        too (the session then has none of that kind). One change, saved, the frames scanned again; Calibration and what
        follows become Outdated. False when the session is not known."""
        if kind not in CALIBRATION_KINDS:
            raise ValueError(f"not a kind of calibration frame: {kind!r}")
        p = self.p
        try:
            s = p.session(session)
        except (KeyError, ValueError):
            return False
        norm = norm_path
        me, here = norm(path), norm(Path(path).parent)
        lst = list(getattr(s, kind))
        if folder is None:
            from ..core.project import list_frames
            listed = {norm(p.abspath(e)) for e in lst}
            notes = {norm(f) for f in s.left_out.get(kind, [])}
            folder = [(str(f), norm(f) in notes) for f in list_frames([str(Path(path).parent)])
                      if here in listed or norm(f) in listed or norm(f) in notes]
        in_folder = [(x, left) for x, left in folder if norm(Path(x).parent) == here]
        if not any(norm(x) == me for x, _left in in_folder):
            in_folder.append((str(path), excluded))
        keep = [x for x, left in in_folder if (norm(x) == me and not excluded) or (norm(x) != me and not left)]
        rest = [e for e in lst if norm(p.abspath(e)) != here and norm(Path(p.abspath(e)).parent) != here]
        with self._edit():
            setattr(s, kind, rest + sorted(keep))
            self._note_left_out(s, kind, [str(path)], excluded)
            p.validate()
        self.reload_sources()
        return True

    def leave_out_calibration(self, frames: list[tuple[int, str, str]],
                              listing: list[tuple[int, str, str, bool]]) -> dict:
        """Leave several calibration frames out of their masters in one change (saved once): ``frames`` (session, kind,
        path); ``listing`` the frames of their folders as the worker listed them (session, kind, path, left out now).
        A session can be left without any frame of a kind (all its "darks" were frames of the sky): ``emptied`` says
        which. Returns {"done": n, "emptied": [(session, kind)]}; nothing done: {"done": 0, "emptied": []}."""
        norm = norm_path
        p = self.p
        groups: dict[tuple[int, str, str], dict[str, str]] = {}
        for sid, kind, path in frames:
            if kind not in CALIBRATION_KINDS:
                raise ValueError(f"not a kind of calibration frame: {kind!r}")
            groups.setdefault((int(sid), kind, norm(Path(path).parent)), {})[norm(path)] = str(path)
        if not groups:
            return {"done": 0, "emptied": []}
        done, touched = 0, set()
        with self._edit():
            for (sid, kind, here), out in groups.items():
                try:
                    s = p.session(sid)
                except (KeyError, ValueError):
                    continue
                lst = list(getattr(s, kind))
                in_folder = [(x, left) for ss, kk, x, left in listing
                             if ss == sid and kk == kind and norm(Path(x).parent) == here]
                known = {norm(x) for x, _left in in_folder}
                in_folder += [(x, False) for n, x in out.items() if n not in known]
                keep = [x for x, left in in_folder if not left and norm(x) not in out]
                done += sum(1 for x, left in in_folder if not left and norm(x) in out)
                rest = [e for e in lst if norm(p.abspath(e)) != here and norm(Path(p.abspath(e)).parent) != here]
                setattr(s, kind, rest + sorted(keep))
                self._note_left_out(s, kind, list(out.values()), True)
                touched.add((sid, kind))
            p.validate()
        emptied = sorted((sid, kind) for sid, kind in touched if not getattr(p.session(sid), kind))
        self.reload_sources()
        return {"done": done, "emptied": emptied}

    def request_selection(self, tmin: float | None = None, fmax: float | None = None):
        """Ask the worker how many frames these thresholds keep (answer: ``selectionChanged``, ``kept_frames``)."""
        p = self.path
        if p is None or not self.ready or self.state(p, "register") != "done":
            return
        t0, f0 = self.thresholds_now()
        m = self.fwhm_median()
        tmin = t0 if tmin is None else tmin
        fmax = f0 if fmax is None else fmax
        want = (p.pid, round(tmin / 100, 6), round(fmax / m if m else fmax, 6), tuple(sorted(self.p.excluded)),
                self._epoch)
        if want == self._sel_want:
            return                                          # asked already: the answer is on its way, or here
        self._sel_want = want
        self.worker.submit(f"model.selection:{id(self)}", _selection_job, self.p, p.pid, want[1], want[2])

    def kept_frames(self) -> int:
        """The count of frames the thresholds keep (the last preview), else the lights of the project."""
        return int(self._sel["kept"]) if self._sel is not None else self.n_lights()

    def selection_total(self) -> int:
        return int(self._sel["total"]) if self._sel is not None else self.n_lights()

    def excluded_mask(self):
        """Which frames (in the order of the registered frames) the last preview leaves out."""
        import numpy as np
        return np.array(self._sel["excluded"] if self._sel is not None else [], bool)

    # ------------------------------------------------------------ sessions and the project's facts
    def info(self) -> dict:
        """The project's facts: from the done inventory result when there is one,
        else from the sessions. Keys: name, target, n_lights, n_darks, n_flats, n_bias, exposure_s, width, height,
        iso_gain, integration_h, camera, fwhm_px, grid."""
        inv = self._facts.inventory
        ses = self.sessions
        return dict(name=self.name, target=self.target,
                    n_lights=int(inv.get("n_lights") or sum(s["lights"] for s in ses)),
                    n_darks=int(inv.get("n_darks") or sum(s["darks"] for s in ses)),
                    n_flats=int(inv.get("n_flats") or sum(s["flats"] for s in ses)),
                    n_bias=int(inv.get("n_bias") or sum(s["bias"] for s in ses)),
                    exposure_s=inv.get("exposure_s"), width=inv.get("width"), height=inv.get("height"),
                    iso_gain=inv.get("iso_gain"), integration_h=inv.get("integration_h"),
                    camera=inv.get("camera", ""), fwhm_px=self._facts.register.get("fwhm_px"),
                    grid=self._facts.stack.get("grid"))

    def image_size(self) -> tuple[int, int] | None:
        """(width, height) of the stack grid, once Stacking is done."""
        g = self._facts.stack.get("grid")
        return (int(g[0]), int(g[1])) if g else None

    def load_sources(self):
        """Frames were added: read them, then the keys again."""
        self.reload_sources()

    def set_stack_mode(self, mode: str) -> bool:
        """"per_session": each session calibrated with its own frames, then all stacked together (issue #43);
        "together": one calibration for all (the projects made before). Calibration and what follows are computed
        again. False when nothing changes."""
        from ..core.project2 import STACK_MODES
        if mode not in STACK_MODES:
            raise ValueError(f"not a stack mode: {mode!r}")
        if mode == self.p.stack_mode:
            return False
        with self._edit():
            self.p.stack_mode = mode
        return True

    @property
    def trunk_params(self) -> dict:
        """The stored settings the pre-processing reads: the current workflow's (each workflow has its own)."""
        p = self.path
        return p.params if p is not None else {}

    def trunk_pending(self) -> list[str]:
        """The pre-processing stages still to compute for the current workflow."""
        p = self.path
        return [x for x in self.pending(p) if x in TRUNK] if p is not None else []

    def trunk_seconds(self) -> float:
        p = self.path
        return sum(self.seconds(x, p) for x in TRUNK) if p is not None else 0.0

    def zone_fraction(self) -> float:
        zx0, zy0, zx1, zy1 = self.test_zone
        return max(1e-4, (zx1 - zx0) * (zy1 - zy0))

    # ------------------------------------------------------------ what a workflow gives in Post
    def slider_infos(self, p: WorkflowView, part: str | None = None) -> list:
        """The Post sliders of a workflow's layer (api.SliderInfo list: key, name, group, ends, available, reason).
        ``part``: None for its Output block's image, else one of its Stars block's ('starless', 'stars', 'combined')."""
        return api.sliders_of(self.p, p.pid, part)

    def slider_state(self, p: WorkflowView, key: str, part: str | None = None):
        """(available, reason why not, options that would unlock it: None when no change of the options would)."""
        info = next((x for x in self.slider_infos(p, part) if x.key == key), None)
        if info is None or info.available:
            return True, "", None
        need = None
        if p.end is None and part is None:
            for tool in ("debruitage",):
                cand = self.with_tool(p.opts, tool)[0]
                if key in self.available_sliders(cand):
                    need = cand
                    break
        return False, _(info.reason), need

    def available_sliders(self, opts: dict) -> list[str]:
        return [s.key for s in api.sliders_of(self.p, self._what_if(opts)) if s.available]

    def unlock_text(self, p: WorkflowView, need: dict) -> str:
        sec, _todo, reused = self.cost_of(need, p.params)
        stars = need["etoiles"] != p.opts["etoiles"]
        dn = need["debruitage"] != p.opts["debruitage"]
        what = (_("Stars/background and Denoise") if stars and dn else _("Stars/background") if stars
                else _("Denoise") if dn else "")
        t = _("Add {what}: +{t}").format(what=what, t=hm(sec))
        if [s for s in reused if s not in set(self.all_stages(p))]:
            t += _(" · rest shared")
        return t

    # ------------------------------------------------------------ the log
    def log(self, template: str, **args):
        """A history line: an English template and its arguments, translated when shown."""
        self.history.append((time.strftime("%H:%M:%S"), template, args))
        if getattr(self, "_naming", False) and self._undo_what and self._undo_what[-1] is None:
            self._undo_what[-1] = (template, args)         # the step's words, until its message says them


# aliases (test_model_surface checks them)
PathDef = WorkflowView
Session = ProjectModel
