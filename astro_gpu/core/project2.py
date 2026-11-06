"""A v2 project: the definition of a processing project with several workflows (project.astroproj).

Layout on disk::

    <project>/
        project.astroproj       the definition (JSON, atomic writes; written only through the API)
        state.json              {"last_keys": {"<wf id>": {"<stage>": "<key>"}}}: previous results of Outdated blocks
        store/<stage>/<key>/    one done result (store/<stage>/<key>.tmp/ while a job runs; store/_trash/)
        logs/                   events.jsonl, run.lock, STOP, runs/<run_id>.plan.json and .log
        cache/  inputs/  export/<name>/
    <scratch_dir or project/scratch>/<key>/<stage>/   large temporaries (the calibrated frame cache)

The definition holds the sessions (nights: lists of folders or files per kind; the engine gets their union), the
frames excluded by hand as (session id, file name) pairs, and the workflows: each one a set of options (core/workflow)
and the parameter values the user stored, kept even when equal to the static default (typing the default is how a
user overrides an automatic value). Paths may be relative to the project in the file; ``abspath`` makes them absolute.

v2 lives beside v1: nothing here writes into a folder holding a v1 ``project.json``. ``load`` raises
``LegacyProject`` on such a folder, and ``from_v1`` builds the v2 definition of it in memory, reading only its
project.json.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from . import workflow as wfl
from .files import write_json
from .project import Inputs, Target

FORMAT = 2
FILE = "project.astroproj"
STATE_FILE = "state.json"
V1_FILE = "project.json"
KINDS = ("lights", "darks", "flats", "bias", "flat_darks")
#: "together": the sessions are one frame list, one calibration (the projects made before issue #43);
#: "per_session": each session calibrated with its own frames, then all stacked together (issue #43)
STACK_MODES = ("together", "per_session")
#: the filters a session may name (mono is planned, not built: a colour sensor has none)
FILTERS = ("", "L", "R", "G", "B", "Ha", "OIII", "SII")


class LegacyProject(Exception):
    """A v1 project folder (project.json, no project.astroproj): open it through ``from_v1`` and the import."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        super().__init__(f"a v1 project (project.json): import it to open it in this version: {self.root}")


@dataclass
class Session:
    """One night (or one card): its frames by kind, each a list of folders or files."""

    id: int
    name: str = ""
    lights: list[str] = field(default_factory=list)
    darks: list[str] = field(default_factory=list)
    flats: list[str] = field(default_factory=list)
    bias: list[str] = field(default_factory=list)
    flat_darks: list[str] = field(default_factory=list)
    filter: str = ""                 # issue #43: the filter of the lights ("" for a colour sensor); mono is planned


@dataclass
class Workflow:
    """One way to process the project: its options, the values the user stored ({stage: {param: value}}), an
    optional early exit, and the workflow it is a variant of (Keep / Discard pending).

    ``breakpoints``: the blocks (engine names: "calibration", "stacking"...) after which a run pauses for the user to
    check the results. It is how the user wants to work, not a way to compute: no key reads it (core/store.py and
    core/workflow.py never see it), so adding or removing one never makes a result Outdated."""

    id: int
    name: str
    options: dict[str, Any] = field(default_factory=lambda: dict(wfl.DEFAULT_OPTIONS))
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    end: str | None = None
    variant_of: int | None = None
    breakpoints: list[str] = field(default_factory=list)
    #: issue #43: an image source, {"path": the image file, "stretched": False}; None: the sessions' frames
    source: dict[str, Any] | None = None
    #: issue #43: the ids of the sessions this workflow stacks (one filter of a monochrome camera: its sessions);
    #: None: every session of the project
    sessions: list[int] | None = None
    #: issue #47: the graph the user built, [{"block", "name", "method", "off"}] after the source (core/workflow.py);
    #: None: a workflow made before, its graph given by its options (workflow.from_options)
    blocks: list[dict[str, Any]] | None = None

    def stored(self, stage: str) -> dict[str, Any]:
        """The values stored for a stage (a copy)."""
        return dict(self.params.get(stage) or {})


@dataclass
class Project2:
    """The definition of a v2 project. ``root`` is where it lives (None for a definition built in memory)."""

    name: str
    description: str = ""
    created: str = ""
    target: Target = field(default_factory=Target)
    scratch_dir: str = ""
    reference_image: str = ""
    devices: dict[str, str] = field(default_factory=lambda: {"main": "", "aux": ""})
    sessions: list[Session] = field(default_factory=list)
    excluded: list[tuple[int, str]] = field(default_factory=list)
    stack_mode: str = "together"
    workflows: list[Workflow] = field(default_factory=list)
    current: int | None = None
    render_bases: dict[str, dict[str, Any]] = field(default_factory=dict)
    post: dict[str, Any] = field(default_factory=lambda: {"layers": []})
    next_workflow_id: int = 1        # ids are never reused, so an undo can bring a deleted workflow back
    format: int = FORMAT
    root: Path | None = field(default=None, compare=False)

    # ------------------------------------------------------------------ paths
    def _root(self) -> Path:
        if self.root is None:
            raise ValueError(f"project {self.name!r} has no folder yet")
        return Path(self.root)

    def abspath(self, path: str | Path) -> str:
        """A path of the definition made absolute (relative ones are relative to the project) and normalised."""
        p = Path(path)
        if not p.is_absolute():
            p = self._root() / p
        return Path(os.path.normpath(p)).as_posix()

    @property
    def store(self) -> Path:
        return self._root() / "store"

    @property
    def logs(self) -> Path:
        return self._root() / "logs"

    @property
    def export(self) -> Path:
        return self._root() / "export"

    @property
    def cache(self) -> Path:
        return self._root() / "cache"

    @property
    def inputs_dir(self) -> Path:
        return self._root() / "inputs"

    @property
    def scratch(self) -> Path:
        return Path(self.abspath(self.scratch_dir)) if self.scratch_dir else self._root() / "scratch"

    def sessions_of(self, wid: int | Workflow | None = None) -> list[Session]:
        """The sessions a workflow stacks (issue #43: a monochrome filter's workflow, its filter's sessions); every
        session for None or a workflow without its own list."""
        w = self.workflow(wid) if wid is not None else None
        if w is None or w.sessions is None:
            return list(self.sessions)
        return [s for s in self.sessions if s.id in w.sessions]

    def is_mono(self, wid: int | Workflow) -> bool:
        """Whether a workflow stacks monochrome frames: every one of its sessions names a filter (a colour camera's
        session names none)."""
        ses = self.sessions_of(wid)
        return bool(ses) and all(s.filter for s in ses)

    def inputs(self, wid: int | Workflow | None = None) -> Inputs:
        """The union of the sessions (of a workflow's), per kind, as absolute paths (one master dark and one master
        flat, as in v1)."""
        out = Inputs()
        for kind in KINDS:
            got: list[str] = getattr(out, kind)
            for s in self.sessions_of(wid):
                for f in getattr(s, kind):
                    a = self.abspath(f)
                    if a not in got:
                        got.append(a)
        return out

    def session_inputs(self, sid: int) -> Inputs:
        """One session's frames, per kind, as absolute paths."""
        s = self.session(sid)
        return Inputs(**{k: [self.abspath(f) for f in getattr(s, k)] for k in KINDS})

    def separate_sessions(self, wid: int | Workflow | None = None) -> list[int]:
        """The ids of the sessions calibrated each on its own (issue #43): every session (of the workflow) when the
        project calibrates them separately and has two or more; else none (one frame list, one calibration)."""
        ses = self.sessions_of(wid)
        return [s.id for s in ses] if self.stack_mode == "per_session" and len(ses) >= 2 else []

    def reference_path(self) -> str:
        """The reference image as an absolute path ('' when there is none)."""
        return self.abspath(self.reference_image) if self.reference_image else ""

    # ------------------------------------------------------------------ sessions and workflows
    def session(self, sid: int) -> Session:
        for s in self.sessions:
            if s.id == sid:
                return s
        raise KeyError(f"no session {sid}")

    def workflow(self, wid: int | Workflow) -> Workflow:
        if isinstance(wid, Workflow):
            return wid
        for w in self.workflows:
            if w.id == wid:
                return w
        raise KeyError(f"no workflow {wid}")

    def add_workflow(self, name: str, options: dict[str, Any], params: dict[str, dict[str, Any]] | None = None,
                     end: str | None = None, variant_of: int | None = None) -> Workflow:
        """A new workflow with a fresh id (ids are never reused); the first one becomes the current one."""
        w = Workflow(self.next_workflow_id, name, wfl.check_options(options), _copy_params(params or {}),
                     wfl.check_end(end), variant_of)
        if variant_of is not None:
            self.workflow(variant_of)
        self.next_workflow_id += 1
        self.workflows.append(w)
        if self.current is None:
            self.current = w.id
        return w

    def remove_workflow(self, wid: int) -> Workflow:
        """Take a workflow out of the definition (its id is not given again); returns it."""
        w = self.workflow(wid)
        self.workflows.remove(w)
        if self.current == wid:
            self.current = self.workflows[0].id if self.workflows else None
        return w

    def chain(self, wid: int | Workflow) -> list[str]:
        """The stages of a workflow, in order (core/workflow.chain on this project's reference image; a graph's in
        its own order, issue #47)."""
        w = self.workflow(wid)
        if w.blocks is not None:
            return wfl.chains().sequence(wfl.sequence_chain(w.blocks, wfl.source_of(w),
                                                            reference=bool(self.reference_image),
                                                            sessions=self.separate_sessions(w)))
        return wfl.chain(w.options, w.end, reference=bool(self.reference_image), sessions=self.separate_sessions(w),
                         image=is_image(w), mono=self.is_mono(w) and w.source is None, fusion=is_fusion(w),
                         stretched=is_stretched(w))

    def set_blocks(self, wid: int | Workflow, blocks: list[dict[str, Any]]) -> None:
        """Give a workflow its graph (checked, issue #47); its options follow (what the code reading them sees)."""
        w = self.workflow(wid)
        w.blocks = wfl.check_blocks(wfl.with_output(blocks, wfl.source_of(w)), wfl.source_of(w))
        w.options = wfl.options_of(w.blocks, wfl.source_of(w))
        w.end = None

    # ------------------------------------------------------------------ checks and JSON
    def validate(self) -> None:
        """Raise ValueError when the definition is not consistent."""
        if not str(self.name).strip():
            raise ValueError("a project needs a name")
        if self.format != FORMAT:
            raise ValueError(f"project format {self.format} is not {FORMAT}")
        sids = [s.id for s in self.sessions]
        if any(not _is_int(i) for i in sids) or len(set(sids)) != len(sids):
            raise ValueError(f"session ids must be distinct integers: {sids}")
        for s in self.sessions:
            for kind in KINDS:
                v = getattr(s, kind)
                if not isinstance(v, list) or not all(isinstance(f, str) and f for f in v):
                    raise ValueError(f"session {s.id}: {kind} must be a list of folders or files")
            if s.filter not in FILTERS:
                raise ValueError(f"session {s.id}: filter {s.filter!r} is not one of {list(FILTERS)}")
        for pair in self.excluded:
            if (not isinstance(pair, tuple) or len(pair) != 2 or not _is_int(pair[0]) or pair[0] not in sids
                    or not isinstance(pair[1], str) or not pair[1]):
                raise ValueError(f"an excluded frame is a (session id, file name) pair of a known session: {pair!r}")
        if self.stack_mode not in STACK_MODES:
            raise ValueError(f"stack mode {self.stack_mode!r} is not in the engine (allowed: {list(STACK_MODES)})")
        wids = [w.id for w in self.workflows]
        if any(not _is_int(i) or i < 1 for i in wids) or len(set(wids)) != len(wids):
            raise ValueError(f"workflow ids must be distinct positive integers: {wids}")
        if wids and self.next_workflow_id <= max(wids):
            raise ValueError(f"next workflow id {self.next_workflow_id} would reuse an id")
        for w in self.workflows:
            if not isinstance(w.name, str) or not w.name.strip():
                raise ValueError(f"workflow {w.id} needs a name")
            wfl.check_options(w.options)
            wfl.check_end(w.end)
            if w.variant_of is not None and (w.variant_of == w.id or w.variant_of not in wids):
                raise ValueError(f"workflow {w.id} is a variant of an unknown workflow {w.variant_of}")
            if not isinstance(w.params, dict) or not all(isinstance(k, str) and isinstance(v, dict)
                                                         for k, v in w.params.items()):
                raise ValueError(f"workflow {w.id}: params must be {{stage: {{param: value}}}}")
            check_breakpoints(w.breakpoints, f"workflow {w.id}")
            check_source(w.source, f"workflow {w.id}")
            if w.blocks is not None:
                try:
                    wfl.check_blocks(w.blocks, wfl.source_of(w))
                except ValueError as e:
                    raise ValueError(f"workflow {w.id}: {e}") from None
            if is_fusion(w) and any(i["pid"] not in wids or i["pid"] == w.id for i in w.source["fusion"]["inputs"]):
                raise ValueError(f"workflow {w.id}: its fusion reads workflows the project does not have")
            st = (w.source or {}).get("fusion", {}).get("stars") if is_fusion(w) else None
            if st and st.get("mode") == "workflow" and (st["pid"] not in wids or st["pid"] == w.id):
                raise ValueError(f"workflow {w.id}: its fusion's stars come from a workflow the project does not have")
            if w.sessions is not None and (not w.sessions or any(x not in sids for x in w.sessions)):
                raise ValueError(f"workflow {w.id}: its sessions {w.sessions} are not sessions of the project")
        if self.current is not None and self.current not in wids:
            raise ValueError(f"current workflow {self.current} does not exist")
        if not isinstance(self.devices, dict):
            raise ValueError("devices must be a dict")

    def to_dict(self) -> dict[str, Any]:
        """The JSON content of project.astroproj."""
        return {"format": self.format, "name": self.name, "description": self.description, "created": self.created,
                "target": asdict(self.target), "scratch_dir": self.scratch_dir,
                "reference_image": self.reference_image, "devices": dict(self.devices),
                "sessions": [asdict(s) for s in self.sessions],
                "excluded": [[sid, name] for sid, name in self.excluded], "stack_mode": self.stack_mode,
                "workflows": [asdict(w) for w in self.workflows], "current": self.current,
                "next_workflow_id": self.next_workflow_id, "render_bases": copy.deepcopy(self.render_bases),
                "post": copy.deepcopy(self.post)}

    @classmethod
    def from_dict(cls, d: dict[str, Any], root: str | Path | None = None) -> Project2:
        """A definition from its JSON content (checked)."""
        if d.get("format") != FORMAT:
            raise ValueError(f"not a format {FORMAT} project definition (format {d.get('format')!r})")
        sessions = [Session(id=s["id"], name=s.get("name", ""), filter=s.get("filter") or "",
                            **{k: list(s.get(k) or []) for k in KINDS})
                    for s in d.get("sessions") or []]
        workflows = [Workflow(id=w["id"], name=w.get("name", ""), options=wfl.check_options(w.get("options") or {}),
                              params=_copy_params(w.get("params") or {}), end=w.get("end"),
                              variant_of=w.get("variant_of"),
                              breakpoints=_breakpoints(w.get("breakpoints")),      # absent in older projects: none
                              source=copy.deepcopy(w.get("source")),
                              sessions=list(w["sessions"]) if w.get("sessions") is not None else None,
                              blocks=copy.deepcopy(w["blocks"]) if w.get("blocks") is not None else None)
                     for w in d.get("workflows") or []]
        nxt = d.get("next_workflow_id") or (max((w.id for w in workflows), default=0) + 1)
        p = cls(name=d.get("name", ""), description=d.get("description", ""), created=d.get("created", ""),
                target=_target(d.get("target") or {}), scratch_dir=d.get("scratch_dir", ""),
                reference_image=d.get("reference_image", ""),
                devices={"main": "", "aux": "", **(d.get("devices") or {})}, sessions=sessions,
                excluded=_excluded(d.get("excluded") or []), stack_mode=d.get("stack_mode", "together"),
                workflows=workflows, current=d.get("current"), render_bases=copy.deepcopy(d.get("render_bases") or {}),
                post=copy.deepcopy(d.get("post") or {"layers": []}), next_workflow_id=nxt,
                root=Path(root) if root is not None else None)
        p.validate()
        return p


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _copy_params(params: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """A deep copy, without the stages that store nothing."""
    return {st: copy.deepcopy(dict(kv)) for st, kv in params.items() if kv}


def check_source(source: Any, where: str = "source") -> dict[str, Any] | None:
    """A workflow's source as it is stored (issue #43): None (the sessions' frames); {"path": a file, "stretched":
    bool} (an image: linear, or already stretched); or {"fusion": {"palette", "inputs": [{"pid", "role",
    "channel"}]}} (other workflows' outputs combined, stages/fusion.py)."""
    if source is None:
        return None
    if isinstance(source, dict) and "fusion" in source:
        from ..stages.fusion import CHANNELS, PALETTES
        f = source["fusion"]
        if set(source) != {"fusion"} or not isinstance(f, dict) or f.get("palette") not in PALETTES:
            raise ValueError(f"{where}: a fusion is {{'fusion': {{'palette': one of {list(PALETTES)}, 'inputs': "
                             f"[...]}}}}, not {source!r}")
        wg = f.get("weights")
        if wg is not None and (not isinstance(wg, dict) or not all(
                isinstance(k, str) and isinstance(v, (int, float)) and 0 <= v <= 5 for k, v in wg.items())):
            raise ValueError(f"{where}: a fusion's weights are {{role: 0 to 5}}, not {wg!r}")
        st = f.get("stars")
        if st is not None and (not isinstance(st, dict) or st.get("mode") not in ("palette", "channel", "workflow")
                               or (st["mode"] == "workflow" and not isinstance(st.get("pid"), int))
                               or (st["mode"] == "channel" and not isinstance(st.get("role"), str))):
            raise ValueError(f"{where}: a fusion's stars are {{'mode': 'palette' | 'channel' (with 'role') | 'workflow' "
                             f"(with 'pid')}}, not {st!r}")
        ins = f.get("inputs")
        if not isinstance(ins, list) or not ins or not all(
                isinstance(i, dict) and isinstance(i.get("pid"), int) and isinstance(i.get("role"), str)
                and i.get("channel", "L") in CHANNELS for i in ins):
            raise ValueError(f"{where}: the fusion's inputs are [{{'pid', 'role', 'channel'}}], not {ins!r}")
        return source
    # path "": an Image block placed from the palette, its file still to choose (it is not run)
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise ValueError(f"{where}: an image source is {{'path': its file, 'stretched': false}}, not {source!r}")
    if set(source) - {"path", "stretched"}:
        raise ValueError(f"{where}: unknown keys of an image source: {sorted(set(source) - {'path', 'stretched'})}")
    return source


def is_fusion(w: Workflow) -> bool:
    """A workflow that combines other workflows' outputs (issue #43)."""
    return isinstance(w.source, dict) and "fusion" in w.source


def is_image(w: Workflow) -> bool:
    """A workflow that starts from an image file (issue #43)."""
    return isinstance(w.source, dict) and "path" in w.source


def is_stretched(w: Workflow) -> bool:
    """A workflow that starts from an image already stretched (not linear)."""
    return is_image(w) and bool(w.source.get("stretched"))


def check_breakpoints(blocks: Any, where: str = "breakpoints") -> list[str]:
    """The breakpoints as they are stored: a list of distinct block names of the engine (core/workflow.BLOCKS)."""
    if not isinstance(blocks, list) or not all(isinstance(b, str) for b in blocks):
        raise ValueError(f"{where}: breakpoints must be a list of block names")
    unknown = [b for b in blocks if b not in wfl.BLOCKS]
    if unknown:
        raise ValueError(f"{where}: no such block to pause after: {unknown} (known: {list(wfl.BLOCKS)})")
    if len(set(blocks)) != len(blocks):
        raise ValueError(f"{where}: a breakpoint is listed twice: {blocks}")
    return blocks


def _breakpoints(raw: Any) -> list[str]:
    """The breakpoints of a workflow from JSON: none when the key is absent (a project saved before breakpoints),
    each block once, in the graph's order."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"breakpoints must be a list of block names, not {raw!r}")
    seen = [b for b in raw if isinstance(b, str)]
    if len(seen) != len(raw):
        raise ValueError(f"breakpoints must be a list of block names: {raw!r}")
    order = list(wfl.BLOCKS)
    return sorted(dict.fromkeys(seen), key=lambda b: order.index(b) if b in order else len(order))


def _target(d: dict[str, Any]) -> Target:
    names = {f.name for f in fields(Target)}
    return Target(**{k: v for k, v in d.items() if k in names})


def _excluded(pairs: list) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"an excluded frame is a (session id, file name) pair: {pair!r}")
        t = (pair[0], pair[1])
        if t not in out:
            out.append(t)
    return out


# ---------------------------------------------------------------------------------------------- files

def _refuse_v1(root: Path) -> None:
    if (root / V1_FILE).exists():
        raise LegacyProject(root)


def load(root: str | Path) -> Project2:
    """Open the v2 project of a folder. A v1 folder (project.json only) raises LegacyProject."""
    root = Path(root)
    f = root / FILE
    if not f.exists():
        if (root / V1_FILE).exists():
            raise LegacyProject(root)
        raise FileNotFoundError(f"not a project folder (no {FILE}): {root}")
    return Project2.from_dict(json.loads(f.read_text(encoding="utf-8")), root)


def save(project: Project2, root: str | Path | None = None) -> Path:
    """Write the definition atomically into ``root`` (default: the project's own folder), which becomes the
    project's folder. Refuses a v1 project folder. Returns the file written."""
    root = Path(root) if root is not None else project._root()
    _refuse_v1(root)
    project.validate()
    root.mkdir(parents=True, exist_ok=True)
    f = root / FILE
    write_json(f, project.to_dict(), indent=2, default=None)
    project.root = root
    return f


def create(root: str | Path, name: str, target: Target | None = None) -> Project2:
    """A new empty v2 project in ``root`` (no sessions, no workflows yet). Refuses an existing project folder."""
    root = Path(root)
    _refuse_v1(root)
    if (root / FILE).exists():
        raise FileExistsError(f"a project already exists in {root}")
    p = Project2(name=name, target=target or Target(), created=datetime.now().isoformat(timespec="seconds"))
    save(p, root)
    return p


def read_state(root: str | Path) -> dict[str, Any]:
    """state.json ({"last_keys": {wf id: {stage: key}}}); empty when absent or unreadable."""
    try:
        d = json.loads((Path(root) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"last_keys": {}}
    d.setdefault("last_keys", {})
    return d


def write_state(root: str | Path, state: dict[str, Any]) -> None:
    """Write state.json atomically (never into a v1 folder)."""
    root = Path(root)
    _refuse_v1(root)
    write_json(root / STATE_FILE, state, indent=2, default=None)


# ---------------------------------------------------------------------------------------------- v1 projects

def v1_options(params: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """The options and the name of the workflow a v1 parameter set gives: denoise pretrained, gradient polynomial
    (the only method of the Gradient block, which v1 did not have), the Stars block ("Full"). ``params`` is not read:
    every v1 project had the stars taken off."""
    return wfl.check_options({"denoise": "pretrained", "stars": "neural", "gradient": "polynomial",
                              "deconvolution": None}), "Full"


def _merge_variant(main: dict[str, dict[str, Any]], changes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Main's values with a v1 variant's changes on top (a change to None puts the value back to automatic)."""
    out = _copy_params(main)
    for st, kv in (changes or {}).items():
        for k, v in (kv or {}).items():
            if v is None:
                out.get(st, {}).pop(k, None)
            else:
                out.setdefault(st, {})[k] = copy.deepcopy(v)
    return _copy_params(out)


def from_v1(root: str | Path) -> Project2:
    """The v2 definition of a v1 project, in memory (``root`` None): one session from its inputs, one workflow from
    main, one per variant (merged values, a variant of main). Only project.json is read; nothing is written. The
    scratch folder is left to the new project's default, so that v2 never writes into v1's scratch."""
    root = Path(root)
    with open(root / V1_FILE, encoding="utf-8") as f:
        d = json.load(f)
    inputs = d.get("inputs") or {}
    session = Session(id=1, name="Session 1", **{k: [str(x) for x in inputs.get(k) or []] for k in KINDS})
    devices = d.get("devices") or {}
    p = Project2(name=d.get("name") or root.name, description=d.get("description", ""), created=d.get("created", ""),
                 target=_target(d.get("target") or {}), reference_image=d.get("reference_image", ""),
                 devices={"main": devices.get("main", ""), "aux": devices.get("aux", "")}, sessions=[session])
    main = d.get("params") or {}
    options, name = v1_options(main)
    first = p.add_workflow(name, options, main)
    active = d.get("active_variant") or ""
    for vname, v in (d.get("variants") or {}).items():
        params = _merge_variant(main, (v or {}).get("params") or {})
        options, _ = v1_options(params)
        w = p.add_workflow(vname, options, params, variant_of=first.id)
        if vname == active:
            p.current = w.id
    p.validate()
    return p
