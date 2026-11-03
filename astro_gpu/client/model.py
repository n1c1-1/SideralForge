"""ProjectModel: the one adapter between the views and the engine API.

Each member is a thin call on ``astro_gpu.api``: the workflows (``WorkflowView``), their stages, keys and states, what a
change costs, the Replace / Try as variant choice, variants, undo and redo, the sessions, the selection thresholds and
the settings of a stage. Edits autosave, debounced 300 ms.

What the GUI thread may do, and what it may not:

* it edits the workflows, and reads keys, states, plans and costs, which the API answers from caches once ``warm`` has
  run: the model submits that to the ApiWorker when a project opens (``ready`` turns True, ``readyChanged`` fires);
* it never scans the disk, loads an array or computes an automatic value: ``describe`` (the settings of a stage), the
  counts of the sessions and the facts of the done results come from the worker. ``params(p, stage)`` answers from a
  cache and asks the worker for what it lacks; ``paramsChanged`` says when it arrives.

Vocabulary. The views use the client's own keys (French words): a workflow's ``opts`` are ``debruitage`` / ``dn``,
``etoiles`` / ``appris``, ``fond`` and ``deconv``; its blocks are ``calibration``, ``alignement``, ``selection``,
``empilement``, ``astrometrie``, ``couleur``, ``deconv``, ``debruitage``, ``etoiles``, ``fond``, ``post`` (blocks.py).
Stage names, settings and the stored ``params`` are the engine's. The model translates at the API's door
(``to_engine`` / ``from_engine``).

A change is a dict: ``{"kind": "param", "stage", "name", "value", "suggested"}`` (value None: back to the suggestion),
``{"kind": "opt", "key", "value"}`` (a method: ``dn``, ``appris``, ``fond``) or ``{"kind": "params", "items": [...]}``
(the Selection thresholds: two settings at once).
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from .. import api
from ..core import workflow as wfl
from ..core.project2 import Workflow, read_state, write_state
from .blocks import COLUMNS, KEY_ENGINE_BLOCK, TOOLS, TRUNK, block_state  # noqa: F401  (TOOLS: test_model_surface reads it here)
from .blocks import ENGINE_BLOCK as ENGINE_KEY
from .i18n import _, hm
from .services.worker import ApiWorker, norm_path

SAVE_DELAY_MS = 300
#: the output stages (the last one of a chain is the workflow's layer)
BY_OUTPUT = ("render_prep", "render_prep_learned", "render_simple")

# ---------------------------------------------------------------------------------------------- vocabulary
#: the starting points of the New workflow dialog, by their keys (dialogs.TEMPLATE_NAME gives their names)
TEMPLATES = {
    "Vierge": dict(debruitage=False, etoiles="garder", appris=False, fond="simple"),
    "Classique + débruitage": dict(debruitage=True, etoiles="garder", appris=False, fond="simple"),
    "Complet": dict(debruitage=True, etoiles="mesurer", appris=False, fond="simple"),
}
DN_SUMMARY = {"affine": "fine-tuned", "generique": "pretrained", "nlbayes": "NL-Bayes"}
CALIBRATION_KINDS = ("darks", "flats", "bias", "flat_darks")    # the session's lists a frame can be left out of
#: seconds of the methods the engine does not have yet (the spec's estimates)
EXTRA_SECONDS = {"denoise_ft": 180.0, "denoise_ai": 45.0, "denoise_cl": 300.0, "deconv": 150.0}
NOT_IN_ENGINE = "Not in the engine yet."
#: the block of the views → the engine's block of ``api.methods``, and the option each method sets
METHOD_BLOCK = {"debruitage": "denoise", "fond": "gradient", "etoiles": "stars", "deconv": "deconvolution"}
METHOD_OPT = {"debruitage": "dn", "fond": "fond", "etoiles": "appris", "deconv": "deconv_m"}
#: the engine's method keys → the views'
METHOD_KEY = {"debruitage": {"n2n": "appris", "finetuned": "affine", "pretrained": "generique", "nlbayes": "nlbayes"},
              "fond": {"polynomial": "simple", "automatic": "auto"},
              "etoiles": {"psf": "psf", "psf_neural": "psf_neural"},
              "deconv": {"rl": "psf", "neural": "neural"}}
#: the toolbox: key of a tool → (engine option, value that adds it, value that removes it)
TOOL_OPTION = {"debruitage": ("denoise", "n2n", None), "etoiles": ("stars", "psf", None),
               "appris": ("stars", "psf_neural", "psf"), "auto": ("gradient", "automatic", "polynomial"),
               "simple": ("gradient", "polynomial", None)}
#: the stages whose settings the summaries of the blocks quote: asked of the worker for each workflow
PREFETCH = ("stack", "denoise", "star_net")


class NotInEngine(ValueError):
    """What was asked is ◇: the engine does not have it yet."""


def to_engine(o: dict) -> dict:
    """The views' options of a workflow as the engine's (core/workflow)."""
    stars = None
    if o.get("etoiles") == "mesurer":
        stars = "psf_neural" if o.get("appris") else "psf"
    return {"denoise": "n2n" if o.get("debruitage") else None, "stars": stars,
            "gradient": "automatic" if o.get("fond") == "auto" else "polynomial", "deconvolution": None}


def from_engine(o: dict) -> dict:
    """The engine's options as the views' (``dn`` is always the engine's Noise2Noise)."""
    return {"debruitage": bool(o.get("denoise")), "dn": "appris", "etoiles": "mesurer" if o.get("stars") else "garder",
            "appris": o.get("stars") == "psf_neural", "fond": "auto" if o.get("gradient") == "automatic" else "simple",
            "deconv": bool(o.get("deconvolution"))}


def methods_of(block: str) -> list[dict]:
    """The methods of a block of the views, as the panel's cards show them: key (the views'), name, short, runs
    (gpu | cpu | both), what, new (◇), stage (the engine's)."""
    return [dict(key=METHOD_KEY[block][m.key], name=m.name, short=m.short, runs=m.runs_on, what=m.what, new=m.new,
                 stage=m.stage, engine_key=m.key) for m in api.methods(METHOD_BLOCK[block])]


def method_of(p: WorkflowView, block: str) -> str:
    if block == "fond":
        return p.opts["fond"]
    if block == "etoiles":
        return "psf_neural" if p.opts.get("appris") else "psf"
    if block == "deconv":
        return p.opts.get("deconv_m", "psf")
    return p.opts.get("dn", "appris")


def method_value(block: str, key: str):
    """The option value that selects a method."""
    if block == "etoiles":
        return key == "psf_neural"
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
    later: list[str] = field(default_factory=list)


#: the reviews kept in state.json (the newest): enough for any project, small for the file
REVIEWED_KEPT = 400


@dataclass
class Facts:
    """What the done results say, read by the worker: the metrics of the stages the views quote."""

    inventory: dict = field(default_factory=dict)
    register: dict = field(default_factory=dict)
    stack: dict = field(default_factory=dict)
    sessions: dict = field(default_factory=dict)         # session id -> {kind: file count}
    reviewed: list = field(default_factory=list)         # the results the user marked as checked (state.json)


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
        counts = {}
        for kind in ("lights", "darks", "flats", "bias", "flat_darks"):
            try:
                counts[kind] = len(list_frames([p.abspath(x) for x in getattr(s, kind)])) if getattr(s, kind) else 0
            except (OSError, ValueError):
                counts[kind] = 0
        f.sessions[s.id] = counts
    if p.root is not None:
        f.reviewed = [str(x) for x in read_state(p.root).get("reviewed") or []]
    return f


def _describe_job(p, wid: int, stage: str):
    key = api.stage_key(p, wid, stage)
    return wid, stage, key, api.describe(p, wid, stage)


def _selection_job(p, wid: int, tmin: float, ratio: float):
    t = api.selection_preview(p, wid, tmin, ratio)
    return wid, round(tmin, 6), round(ratio, 6), t.kept, t.total, [r.status != "" for r in t.rows]


def _sig(rows) -> tuple:
    return tuple((r.name, repr(r.value), repr(r.suggested), r.origin, r.reason, repr(r.reason_args), r.drift,
                  r.locked) for r in rows or [])


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

    def __init__(self, project, worker: ApiWorker | None = None, *, readonly: bool = False,
                 history: list | None = None, gpu: bool = True, device: str | None = None):
        super().__init__()
        self.p = project                                   # the definition (api.Project2): the source of truth
        self.root = Path(project.root) if project.root is not None else None
        self.readonly = readonly                           # never write the project (tests, fixtures)
        self._own_worker = worker is None
        self.worker = worker or ApiWorker(self)
        self.worker.done.connect(self._worker_done)
        self.worker.failed.connect(self._worker_failed)
        self.paths: list[WorkflowView] = []
        self.current = 0
        self.sel_step = "sources"
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
        self._depth = 0
        self._dirty = False
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self.flush)
        self._rebuild_views()
        if self.paths:
            self.current = max(0, next((i for i, v in enumerate(self.paths) if v.pid == project.current), 0))
            self.sel_step = "empilement"
        self._start_warm()

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
                            iso=self._facts.inventory.get("iso_gain"), temp=None, filter="OSC", rotation=0.0,
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
        """The sessions or the frames changed (Import folder, a new session): the definition is saved, the frames are
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

    def close(self):
        """Save what is pending and let go of the worker (the project is closed)."""
        self.flush()
        self.worker.cancel(f"model.warm:{id(self)}")
        for sig, slot in ((self.worker.done, self._worker_done), (self.worker.failed, self._worker_failed)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        if self._own_worker:
            self.worker.stop()

    # ------------------------------------------------------------ saving
    def flush(self):
        """Write the definition now (the edits are saved 300 ms after the last one)."""
        self._save_timer.stop()
        if self._dirty and not self.readonly and self.root is not None:
            if self.path is not None:
                self.p.current = self.path.pid
            api.save(self.p)
        self._dirty = False

    def edited(self):
        """The definition changed outside the workflows (the sessions, the frames excluded by hand, the Post
        composition): it is saved 300 ms after the last change, as any edit."""
        self._dirty = True
        if not self.readonly:
            self._save_timer.start()

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
            out = next((s for s in reversed(stages) if s in BY_OUTPUT), None)
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
            for block in ("debruitage", "etoiles", "fond"):
                try:
                    for m in methods_of(block):
                        for dev in ("gpu", "cpu"):                 # the card says both: GPU and CPU
                            self.method_seconds(v, block, m["key"], dev)
                except (ValueError, KeyError):
                    pass
            for k in TOOLBOX:
                if k != "deconv" and not self.has_tool(v.opts, k):
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
                self._prune_breakpoints()
                if self._snapshot() != before:
                    self._undo.append(before)
                    del self._undo[:-200]
                    self._redo.clear()
                    self.edited()
                elif self._extra() != extra:
                    self.edited()                           # not for Undo: the frames, not the workflows
                self._sync()
                self.pathsChanged.emit()

    def _snapshot(self):
        return (api.workflows(self.p), self.p.current, self.p.stack_mode)

    def _extra(self):
        """What an edit may change besides the workflows: the sessions and the frames excluded by hand."""
        return (sorted(tuple(x) for x in self.p.excluded),
                [(s.id, s.name, *(tuple(getattr(s, k)) for k in ("lights", "darks", "flats", "bias", "flat_darks")))
                 for s in self.p.sessions])

    # ------------------------------------------------------------ undo and redo
    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        return self._restore(self._undo, self._redo)

    def redo(self) -> bool:
        return self._restore(self._redo, self._undo)

    def _restore(self, src: list, dst: list) -> bool:
        """Put the workflows back as an earlier snapshot had them (computed results are never removed)."""
        if not src:
            return False
        dst.append(self._snapshot())
        ws, current, mode = src.pop()
        api.set_workflows(self.p, ws, current)
        self.p.stack_mode = mode
        self.pending_change = None
        self.edited()
        self._sync()
        if self.paths:
            self.current = next((i for i, v in enumerate(self.paths) if v.pid == self.p.current), 0)
            keys = self.block_keys(self.path)
            if self.sel_step not in ("sources", "final") and self.sel_step not in keys:
                self.sel_step = keys[-1] if keys else "sources"
        self.pathsChanged.emit()
        self.currentChanged.emit()
        return True

    # ------------------------------------------------------------ stages, keys, states
    def all_stages(self, p: WorkflowView | None = None) -> list[str]:
        p = p or self.path
        return list(self._chains.get(p.pid, [])) if p is not None else []

    def path_stages(self, p: WorkflowView | None = None) -> list[str]:
        return [s for s in self.all_stages(p) if s not in TRUNK]

    def block_keys(self, p: WorkflowView) -> list[str]:
        """The views' block keys of a workflow, in order (the frames and Post excluded; Output last)."""
        return [ENGINE_KEY[b.block] for b in self._blocks.get(p.pid, []) if b.block in ENGINE_KEY]

    def block_key(self, p: WorkflowView, key: str):
        """What two workflows must have equal to share a block (with everything before it): the keys of its stages
        (the engine's ``blocks``); Selection is the thresholds (it has no stage of its own); Output has the
        workflow's id."""
        if key == "selection":
            st = p.params.get("stack") or {}
            return ("selection", st.get("min_transparency"), st.get("max_fwhm_ratio"),
                    st.get("selection", "min_weight"), st.get("selection_transparency", "on"))
        ref = next((b for b in self._blocks.get(p.pid, []) if ENGINE_KEY.get(b.block) == key), None)
        return (key, ref.key if ref is not None else ())

    def title(self, stage: str) -> str:
        names = {"inventory": "Inventory", "calibrate": "Calibration", "register": "Registration",
                 "stack": "Stacking", "astrometry": "Astrometry", "reference": "Reference image",
                 "catalogs": "Gaia catalogues", "surveys": "H-alpha survey", "color": "Color", "denoise": "Denoise",
                 "optics": "Optics", "stars": "Star measurement", "star_net": "Neural star removal",
                 "starless": "Starless image", "background": "Sky level", "gradient": "Sky gradient",
                 "render_prep": "Output prep", "render_prep_learned": "Output prep (neural removal)",
                 "render_simple": "Output prep (simple)", "deconv": "Deconvolution"}
        return _(names.get(stage, stage))

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
        tool = {"fond": {"auto": "auto", "simple": "simple"}, "etoiles": {"psf": "etoiles", "psf_neural": "appris"},
                "debruitage": {"appris": "debruitage"}}.get(block, {}).get(method)
        try:
            opts = self.with_tool(p.opts, tool)[0] if tool else dict(p.opts)      # the options the method needs
        except wfl.OptionRefused:
            return None
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
            w = self._what_if(opts, p.params)
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
            self._cache[k] = api.plan_to_block(self.p, p.pid, KEY_ENGINE_BLOCK[key], self.device, self._history_est)
        return self._cache[k]

    # ------------------------------------------------------------ breakpoints: pause for the user to check
    # A breakpoint is a block of one workflow ("Pause here for review"): a run stops after it, and goes on only when
    # the user says so (Continue). It lives in the definition (``Workflow.breakpoints``, engine names) and is no part of
    # any key. What is reviewed is tied to the block's results: a fingerprint of its stage keys, kept in state.json.
    def has_breakpoint(self, p: WorkflowView, key: str) -> bool:
        """Whether this workflow pauses after this block (a block of the views)."""
        eb = KEY_ENGINE_BLOCK.get(key)
        return eb is not None and eb in self.p.workflow(p.pid).breakpoints

    def breakpoints(self, p: WorkflowView) -> list[str]:
        """The blocks (of the views) this workflow pauses after, in the graph's order; only those it has."""
        have = {ENGINE_KEY[b] for b in self.p.workflow(p.pid).breakpoints if b in ENGINE_KEY}
        return [k for k in self.block_keys(p) if k in have]

    def set_breakpoint(self, paths, key: str, on: bool = True):
        """Pause after this block in these workflows (or no longer). One change for Undo; saved with the project."""
        eb = KEY_ENGINE_BLOCK.get(key)
        if eb is None:
            raise ValueError(f"no breakpoint on {key!r}: it is not a block of the engine")
        order = list(wfl.BLOCKS)
        with self._edit():
            for q in paths:
                w = self.p.workflow(q.pid)
                have = set(w.breakpoints)
                w.breakpoints = sorted(have | {eb} if on else have - {eb}, key=order.index)

    def _prune_breakpoints(self):
        """A block that left a workflow takes its breakpoint with it (it would otherwise come back with the block)."""
        if not self.ready:
            return
        for w in self.p.workflows:
            if not w.breakpoints:
                continue
            try:
                inside = {b.block for b in api.blocks(self.p, w.id)}
            except (ValueError, KeyError, OSError):
                continue
            if any(b not in inside for b in w.breakpoints):
                w.breakpoints = [b for b in w.breakpoints if b in inside]

    def review_key(self, p: WorkflowView, key: str) -> str:
        """What a review is tied to: the block's results, that is the keys of its stages (the engine's ``blocks``);
        for Selection, which has no stage, the Registration it follows, the thresholds and the frames excluded by hand."""
        k = ("rk", p.pid, key)
        if k not in self._cache:
            def ref(view_key):
                return next((b for b in self._blocks.get(p.pid, []) if ENGINE_KEY.get(b.block) == view_key), None)
            b = ref(key)
            parts: list = [KEY_ENGINE_BLOCK.get(key, key), list(b.key) if b is not None else []]
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
        return "paused" if key == first else "pending"

    def pause_point(self, only: WorkflowView | None = None, upto: str | None = None) -> Pause | None:
        """The review a run stops at: the first block, in the graph's order, that has a breakpoint whose results are
        not reviewed, in any workflow (``only``: in this one, up to the block ``upto``). None when the run goes
        through."""
        if not self.ready:
            return None
        found: list[tuple[int, int, WorkflowView, str]] = []
        later: set[str] = set()
        for i, q in enumerate(self.paths if only is None else [only]):
            keys = self.block_keys(q)
            reach = keys[:keys.index(upto) + 1] if upto in keys else keys
            todo = [k for k in self.breakpoints(q) if k in reach and not self.is_reviewed(q, k)]
            if todo:
                found.append((COLUMNS.index(todo[0]), i, q, todo[0]))
                later |= set(todo[1:])
        if not found:
            return None
        col, _i, _q, block = min(found, key=lambda t: t[:2])
        pids = [q.pid for _c, _j, q, k in found if k == block]
        later |= {k for _c, _j, _q, k in found if k != block}
        mine = [q for q in self.paths if q.pid in pids]
        return Pause(block, pids, all(block_state(self, q, block) == "done" for q in mine),
                     sorted((k for k in later if COLUMNS.index(k) > col), key=COLUMNS.index))

    def mark_reviewed(self, pause: Pause):
        """Continue: the results the user has looked at are marked as checked. Only a block that is computed now is
        marked, with the results it has now: if the user changed something that made it Outdated meanwhile, the
        breakpoint applies again to the new results."""
        for pid in pause.pids:
            q = self.view(pid)
            if q is not None and self.ready and block_state(self, q, pause.block) == "done":
                k = self.review_key(q, pause.block)
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
        """Run all, with the breakpoints: up to the first review not yet done (every workflow that pauses there), else
        everything. Returns the plan and the pause it ends at (None: the whole project)."""
        pause = self.pause_point()
        return (self.plan_all(), None) if pause is None else (self._plan_pause(pause), pause)

    def plan_run_to(self, p: WorkflowView, key: str) -> tuple[api.Plan, Pause | None]:
        """Run to here, with the breakpoints: the block, or the first review of this workflow before it."""
        pause = self.pause_point(only=p, upto=key)
        return (self.plan_to_block(p, key), None) if pause is None else (self._plan_pause(pause), pause)

    def _plan_pause(self, pause: Pause) -> api.Plan:
        """The plans to the block of every workflow that pauses there, as one plan: a job two plans share is kept once
        (the jobs of a plan come after those they read, so the concatenation stays in order)."""
        plans = [self.plan_to_block(q, pause.block) for q in self.paths if q.pid in pause.pids]
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
        return {"debruitage": bool(o.get("debruitage")), "etoiles": o.get("etoiles") == "mesurer",
                "appris": bool(o.get("appris")), "auto": o.get("fond") == "auto",
                "simple": o.get("fond") == "simple", "deconv": bool(o.get("deconv"))}[key]

    @staticmethod
    def with_tool(o: dict, key: str) -> tuple[dict, list[str]]:
        """The options with this tool added, and what came with it (the notes are English texts to translate)."""
        if key == "deconv":
            return dict(o), [NOT_IN_ENGINE]
        opt, add, _rm = TOOL_OPTION[key]
        new, notes = wfl.set_option(to_engine(o), opt, add)
        return {**o, **from_engine(new)}, list(notes)

    @staticmethod
    def without_tool(o: dict, key: str) -> tuple[dict | None, list[str]]:
        """The options with this tool removed and what went with it, or (None, [reason]) when it cannot go."""
        if key == "deconv":
            return dict(o), []
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
    def unique_name(self, base: str) -> str:
        names = {p.name for p in self.paths}
        if base not in names:
            return base
        n = 2
        while f"{base} {n}" in names:
            n += 1
        return f"{base} {n}"

    def add_path(self, name: str, opts: dict, params: dict | None = None) -> WorkflowView:
        """A new workflow. Without settings of its own it joins the current one's pre-processing (the settings of
        Calibration to Stacking are copied), so that the two share those blocks."""
        params = copy.deepcopy(params or {})
        if not params and self.path is not None:
            params = {s: dict(v) for s, v in self.path.params.items() if s in TRUNK}
        with self._edit():
            w = api.add_workflow(self.p, name, wfl.check_options(to_engine(opts)), params)
        return self.view(w.id)

    def batch(self):
        """Several edits that Undo takes back as one: ``with model.batch(): ...``."""
        return self._edit()

    def duplicate(self, src: WorkflowView, name: str) -> WorkflowView:
        """A copy of a workflow: its options, settings and exit."""
        with self._edit():
            w = api.add_workflow(self.p, self.unique_name(name), None, None, copy_of=src.pid)
            self.p.workflow(w.id).breakpoints = list(self.p.workflow(src.pid).breakpoints)
        return self.view(w.id)

    def rename(self, pid: int, name: str):
        with self._edit():
            api.rename_workflow(self.p, pid, name)

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
            if v != "appris":
                raise NotInEngine(NOT_IN_ENGINE)
            return [{"kind": "opt", "key": "denoise", "value": "n2n"}]
        if key == "appris":
            return [{"kind": "opt", "key": "stars", "value": "psf_neural" if v else "psf"}]
        if key == "fond":
            return [{"kind": "opt", "key": "gradient", "value": "automatic" if v == "auto" else "polynomial"}]
        if key == "debruitage":
            return [{"kind": "opt", "key": "denoise", "value": "n2n" if v else None}]
        if key == "etoiles":
            return [{"kind": "opt", "key": "stars", "value": "psf" if v == "mesurer" else None}]
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
            self.p.workflow(w.id).breakpoints = list(self.p.workflow(p.pid).breakpoints)
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

    def set_calibration_excluded(self, session: int, kind: str, path: str, excluded: bool,
                                 folder: list[tuple[str, bool]] | None = None) -> bool:
        """Leave a calibration frame out of its master, or take it back (``kind``: darks, flats, bias or flat_darks).
        The engine's lists take folders and files, so the session lists the frames of that frame's folder one by one,
        without it; when every frame of the folder is in again, the folder itself is listed, as it was. ``folder``: the
        frames of that folder and whether each is left out now, as the worker listed them
        (``services/inspect.browse_group``); None: the folder is listed here (a directory listing in the GUI thread).
        One change, saved, the frames scanned again; Calibration and what follows become Outdated. False when nothing is
        done: the last frame of a folder stays (remove the folder instead), or the session is not known."""
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
            folder = [(str(f), not (here in listed or norm(f) in listed)) for f in list_frames([str(Path(path).parent)])]
        in_folder = [(x, left) for x, left in folder if norm(Path(x).parent) == here]
        if not any(norm(x) == me for x, _left in in_folder):
            in_folder.append((str(path), excluded))
        keep = [x for x, left in in_folder if (norm(x) == me and not excluded) or (norm(x) != me and not left)]
        if not keep:
            return False
        rest = [e for e in lst if norm(p.abspath(e)) != here and norm(Path(p.abspath(e)).parent) != here]
        original = next((e for e in lst if norm(p.abspath(e)) == here), None)
        whole = len(keep) == len(in_folder)
        with self._edit():
            setattr(s, kind, rest + ([original or str(Path(path).parent)] if whole else sorted(keep)))
            p.validate()
        self.reload_sources()
        return True

    def leave_out_calibration(self, frames: list[tuple[int, str, str]],
                              listing: list[tuple[int, str, str, bool]]) -> dict:
        """Leave several calibration frames out of their masters in one change (saved once): ``frames`` (session, kind,
        path); ``listing`` the frames of their folders as the worker listed them (session, kind, path, left out now).
        A folder whose every frame goes is taken out of the session, so a session can be left without any frame of a
        kind (all its "darks" were frames of the sky): ``emptied`` says which. Returns {"done": n, "emptied":
        [(session, kind)]}; nothing done: {"done": 0, "emptied": []}."""
        norm = norm_path
        p = self.p
        groups: dict[tuple[int, str, str], set[str]] = {}
        for sid, kind, path in frames:
            if kind not in CALIBRATION_KINDS:
                raise ValueError(f"not a kind of calibration frame: {kind!r}")
            groups.setdefault((int(sid), kind, norm(Path(path).parent)), set()).add(norm(path))
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
                in_folder += [(x, False) for x in out if x not in known]
                keep = [x for x, left in in_folder if not left and norm(x) not in out]
                done += sum(1 for x, left in in_folder if not left and norm(x) in out)
                rest = [e for e in lst if norm(p.abspath(e)) != here and norm(Path(p.abspath(e)).parent) != here]
                original = next((e for e in lst if norm(p.abspath(e)) == here), None)
                whole = len(keep) == len(in_folder)
                setattr(s, kind, rest + (([original or str(Path(next(iter(out))).parent)] if whole else sorted(keep))
                                         if keep else []))
                touched.add((sid, kind))
            p.validate()
        emptied = sorted((sid, kind) for sid, kind in touched if not getattr(p.session(sid), kind))
        self.reload_sources()
        return {"done": done, "emptied": emptied}

    def add_calibration_folder(self, session: int, kind: str, folder: str) -> bool:
        """A folder of calibration frames (darks, flats, bias, flat darks) added to a session: saved, the frames read
        again, Calibration and what follows Outdated. False when the session is not known or lists it already."""
        if kind not in CALIBRATION_KINDS:
            raise ValueError(f"not a kind of calibration frame: {kind!r}")
        try:
            s = self.p.session(session)
        except (KeyError, ValueError):
            return False
        lst = list(getattr(s, kind))
        if any(norm_path(self.p.abspath(e)) == norm_path(folder) for e in lst):
            return False
        with self._edit():
            setattr(s, kind, lst + [Path(folder).as_posix()])
            self.p.validate()
        self.reload_sources()
        return True

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
        """Frames were added (Import folder): read them, then the keys again."""
        self.reload_sources()

    def add_session(self, *a, **k) -> bool:
        """Sessions are added through the Import folder dialog (api.add_frames, then ``reload_sources``)."""
        return False

    def set_stack_mode(self, mode: str) -> bool:
        """All sessions together is the only mode of this version (one stack per session is ◇)."""
        return mode == self.p.stack_mode

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
    def slider_infos(self, p: WorkflowView) -> list:
        """The Post sliders of a workflow's layer (api.SliderInfo list: key, name, group, ends, available, reason)."""
        return api.sliders_of(self.p, p.pid)

    def slider_state(self, p: WorkflowView, key: str):
        """(available, reason why not, options that would unlock it: None when no change of the options would)."""
        info = next((x for x in self.slider_infos(p) if x.key == key), None)
        if info is None or info.available:
            return True, "", None
        need = None
        if p.end is None:
            for tool in ("etoiles", "debruitage"):
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
        what = _("Stars and Denoise") if stars and dn else _("Stars") if stars else _("Denoise") if dn else ""
        t = _("Add {what} to the workflow: +{t}").format(what=what, t=hm(sec))
        if [s for s in reused if s not in set(self.all_stages(p))]:
            t += _(", the rest is already computed by another workflow")
        return t

    # ------------------------------------------------------------ the log
    def log(self, template: str, **args):
        """A history line: an English template and its arguments, translated when shown."""
        self.history.append((time.strftime("%H:%M:%S"), template, args))


# aliases (test_model_surface checks them)
PathDef = WorkflowView
Session = ProjectModel
