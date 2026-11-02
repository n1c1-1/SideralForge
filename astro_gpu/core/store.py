"""The store of a v2 project: stage results addressed by a key, and the project a job's stage code sees.

A done stage result lives in ``store/<stage>/<key>/`` (DESIGN §1.1, §1.3). Its key hashes everything that decides
the result, and nothing else::

    key(wf, s) = sha1(canonical_json({"s": s, "v": version, "p": key_params(wf, s),
                                      "u": {d: key(wf, d) for d in deps_in_chain(wf, s)},
                                      "x": project_part(s)}))[:16]

``key_params`` are the values the workflow stored, with those its options imply and the excluded frames on top,
validated through the stage's Params (2, 2.0 and "2" hash the same; an invalid value is a ValueError naming the
setting), and path settings replaced by their file (name, size, mtime). The rule on defaults:

- a field with an automatic rule (``AUTO_FIELDS``) is never dropped: a stored value equal to the static default
  overrides the automatic value, so it may be another result;
- a field without one is dropped at its static default: the effective value is the same;
- a field whose default changed (``KEY_BASELINE``: its previous default) is in the key with its effective value,
  stored or not, unless that value is the previous default: unset, it gets a new key (the new default's result);
  stored at the previous default, it keeps the key it always had. A setting removed from a stage (``RETIRED``) is
  ignored where a definition still stores it; one whose last value must stay in the keys (``KEY_CONSTANTS``) is
  added with that value, so the results computed with it keep their keys.

``version`` is the stage's, or what its ``key_version(values)`` makes of the definition's settings when it has one: a
part of a stage revised on its own changes the keys of the definitions that use it only (the stack's Classic part).

Hence a setting added to a stage later, whose static default keeps the previous behaviour and which has no automatic
rule at that default, never enters the key of a definition that leaves it at its default, stored or not: every key
computed before it existed stays the same (tests/test_stacking_v2.py). That is how a new option goes in beside the old
behaviour without a version bump.

Automatic values are not in the key: they follow from the upstream results, which the upstream keys stand for.
Settings that never change a result (worker counts, the training GPU) are left out. The key is the same in every
process (canonical JSON, no set or hash order).

The project parts read the disk (each session's frame files, the reference image, path settings): ``InputScan``
caches them. A scan stats every frame: keep it out of the GUI thread.

``StoreProject`` runs unchanged stage code, and v1's ``Pipeline._run_stage``, for one job: its own stage writes into
``<key>.tmp`` (the runner renames it on success), every ancestor reads its done key folder, and any other stage
resolves to ``store/_absent/<stage>``, which is never created (``Context.has`` is false there).
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import workflow as wfl
from .params import StageParams
from .project import Inputs, ProjectConfig, list_frames
from .project2 import KINDS, Project2, Workflow
from .workflow import Chains

STORE = "store"
TMP = ".tmp"                 # store/<stage>/<key>.tmp/: a job in progress, never done
TRASH = "_trash"             # done folders replaced by a re-run
ABSENT = "_absent"           # where the stages a chain does not have resolve; never created
KEY_LEN = 16
_KEY = re.compile(r"^[0-9a-f]{16}$")

#: settings that never change a result: left out of the key (a job still gets them)
NON_KEY: dict[str, tuple[str, ...]] = {"calibrate": ("workers",), "register": ("workers",), "star_net": ("device",)}
#: settings that name a file, keyed by that file. stages/v2.py declares them; this is the design's list until it does
PATH_FIELDS: dict[str, tuple[str, ...]] = {"starless": ("learned_stars",), "denoise": ("reuse_model",)}
#: a 'Learned star layer' value may carry the star_net run that made it: '<path> (run <id>)' (star_net.layer_file)
_RUN_TAG = re.compile(r"^(.*?) \(run [0-9a-fA-F]*\)$")


def canonical_json(obj: Any) -> str:
    """The one JSON text of a value that gets hashed: sorted keys, no spaces, ASCII."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def mtime_key(mtime_ns: int) -> int:
    """A file's mtime as a key holds it: in seconds, rounded UP to an even second. exFAT keeps modification times to
    2 s and rounds a copied file's time up to the next even second, so a copy from NTFS to an exFAT disk gets a
    different time; rounded up to the even second, both give the same. A time already on an even second (every file
    written on exFAT) keeps its value, so the keys already computed there do not move."""
    return -(-mtime_ns // 2_000_000_000) * 2


def file_id(path: str | Path) -> list:
    """What stands for a file in a key: [name, size, mtime in seconds rounded up to even (``mtime_key``)], or
    [name, None, None] when it does not exist. The folder is left out, so a copy elsewhere (on NTFS or exFAT) that
    keeps the date keeps the key."""
    p = Path(path)
    name = os.path.normcase(p.name)
    try:
        st = p.stat()
    except OSError:
        return [name, None, None]
    return [name, st.st_size, mtime_key(st.st_mtime_ns)]


def _absolute(project: Project2, path: str | Path) -> str:
    """A path of the definition, absolute (relative to the project) and normalised, with '/'."""
    return Path(os.path.abspath(project.abspath(path))).as_posix()


def _split_tag(value: str) -> tuple[str, str]:
    """('<path>', ' (run <id>)') of a path setting; the tag is '' when there is none."""
    v = value.strip()
    m = _RUN_TAG.match(v)
    return (m.group(1), v[len(m.group(1)):]) if m else (v, "")


# ---------------------------------------------------------------------------------------------- rules

class KeyRules:
    """What the key needs to know of the stages: their graph (``Chains``), the fields an ``auto()`` can set, the
    path fields, the fields that never change a result, and the values workflow options imply.
    ``auto_fields`` None counts every field as automatic: nothing is ever dropped (the safe side)."""

    def __init__(self, chains: Chains, auto_fields: Mapping[str, Iterable[str]] | None = None,
                 path_fields: Mapping[str, Iterable[str]] | None = None,
                 non_key: Mapping[str, Iterable[str]] | None = None,
                 implied: Callable[[dict[str, Any], str], dict[str, Any]] = wfl.implied,
                 baseline: Mapping[str, Mapping[str, Any]] | None = None,
                 retired: Mapping[str, Iterable[str]] | None = None,
                 constants: Mapping[str, Mapping[str, Any]] | None = None):
        self.chains = chains
        self._auto = None if auto_fields is None else {s: frozenset(f) for s, f in auto_fields.items()}
        self._baseline = {s: dict(d) for s, d in (baseline or {}).items()}
        self._retired = {s: frozenset(f) for s, f in (retired or {}).items()}
        self._constants = {s: dict(d) for s, d in (constants or {}).items()}
        self._paths = {s: frozenset(f) for s, f in (PATH_FIELDS if path_fields is None else path_fields).items()}
        self._non_key = {s: frozenset(f) for s, f in (NON_KEY if non_key is None else non_key).items()}
        self.implied = implied
        self._defaults: dict[str, dict[str, Any]] = {}

    def params(self, stage: str) -> type[StageParams]:
        return self.chains.stages[stage].Params

    def version(self, stage: str, values: Mapping[str, Any] | None = None) -> str:
        """The stage's version in a key: its ``version``; with the definition's validated ``values``, the stage's
        ``key_version(values)`` when it has one (a part of the stage revised on its own, keyed only where the settings
        use it: the stack's Classic part)."""
        st = self.chains.stages[stage]
        kv = getattr(st, "key_version", None)
        return str(kv(dict(values)) if callable(kv) and values is not None else st.version)

    def auto_fields(self, stage: str) -> frozenset[str]:
        """The fields an ``auto()`` can set (stages/v2.AUTO_FIELDS); without that table, every field (the safe
        side)."""
        if self._auto is None:
            return frozenset(self.params(stage).model_fields)
        return self._auto.get(stage, frozenset())

    def baseline(self, stage: str) -> dict[str, Any]:
        """{field: its previous default} of the fields whose default changed (stages/v2.KEY_BASELINE): the key holds
        their effective value, stored or not, unless it is that previous default."""
        return self._baseline.get(stage, {})

    def retired(self, stage: str) -> frozenset[str]:
        """The settings removed from the stage (stages/v2.RETIRED): a stored one is ignored."""
        return self._retired.get(stage, frozenset())

    def constants(self, stage: str) -> dict[str, Any]:
        """{field: value} of the removed settings whose last value stays in every key (stages/v2.KEY_CONSTANTS)."""
        return self._constants.get(stage, {})

    def path_fields(self, stage: str) -> frozenset[str]:
        return self._paths.get(stage, frozenset())

    def non_key(self, stage: str) -> frozenset[str]:
        return self._non_key.get(stage, frozenset())

    def defaults(self, stage: str) -> dict[str, Any]:
        """The static defaults of a stage, in their JSON form."""
        if stage not in self._defaults:
            self._defaults[stage] = self.params(stage)().model_dump(mode="json")
        return self._defaults[stage]

    def injects_exclude(self, stage: str) -> bool:
        """Whether the definition's excluded frames reach this stage (``stack.exclude``)."""
        return stage == "stack" and "exclude" in self.params(stage).model_fields


_RULES: list[KeyRules] = []


def key_rules() -> KeyRules:
    """The rules of the v2 stages: ``workflow.chains()`` with AUTO_FIELDS and PATH_FIELDS of stages/v2.py. Until
    that module exists, every field counts as automatic and PATH_FIELDS is the design's list."""
    if not _RULES:
        v2 = _v2_module()
        _RULES.append(KeyRules(wfl.chains(), getattr(v2, "AUTO_FIELDS", None), getattr(v2, "PATH_FIELDS", None),
                               baseline=getattr(v2, "KEY_BASELINE", None), retired=getattr(v2, "RETIRED", None),
                               constants=getattr(v2, "KEY_CONSTANTS", None)))
    return _RULES[0]


def _v2_module() -> Any:
    name = importlib.util.resolve_name("..stages.v2", __package__)
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise
        return None


# ---------------------------------------------------------------------------------------------- disk parts

class InputScan:
    """The parts of the keys that read the disk, cached: each session's frame files (by its folder and file lists)
    and file ids (the reference image, path settings). Call ``forget`` when files may have changed."""

    def __init__(self) -> None:
        self._sessions: dict[tuple, dict[str, list[tuple[Path, int]]]] = {}
        self._files: dict[str, list] = {}

    def forget(self) -> None:
        self._sessions.clear()
        self._files.clear()

    def session_frames(self, project: Project2, session: Any) -> dict[str, list[tuple[Path, int]]]:
        """{kind: [(file, size)]} of one session, as list_frames finds them."""
        entries = tuple(tuple(_absolute(project, f) for f in getattr(session, k)) for k in KINDS)
        got = self._sessions.get(entries)
        if got is None:
            got = {}
            for kind, paths in zip(KINDS, entries):
                files = []
                for f in list_frames(list(paths)):
                    try:
                        files.append((f, f.stat().st_size))
                    except OSError:              # gone since the listing
                        pass
                got[kind] = files
            self._sessions[entries] = got
        return got

    def frames(self, project: Project2) -> list[list]:
        """inventory's project part: [kind, file name, size] of each frame of the union of the sessions, per kind, in
        the order the inventory reads them (list_frames: by full path). Names and sizes only, so copied raws keep
        the key; ordered, because the half-stack blocks follow the order."""
        out: list[list] = []
        for kind in KINDS:
            sizes: dict[Path, int] = {}
            for s in project.sessions:
                for f, n in self.session_frames(project, s)[kind]:
                    sizes.setdefault(f, n)
            out += [[kind, f.name, sizes[f]] for f in sorted(sizes)]
        return out

    def excluded_paths(self, project: Project2) -> list[str]:
        """The files of the excluded (session, name) pairs, written as the inventory writes them (``stack.exclude``)."""
        out: set[str] = set()
        for sid, name in project.excluded:
            frames = self.session_frames(project, project.session(sid))["lights"]
            out.update(str(f) for f, _ in frames if f.name == name)
        return sorted(out)

    def file_id(self, path: str) -> list:
        if path not in self._files:
            self._files[path] = file_id(path)
        return self._files[path]


# ---------------------------------------------------------------------------------------------- keys

class Keys:
    """The stage keys of one state of a project definition. Build a new one after an edit (it is cheap; pass the
    same ``InputScan`` to keep the disk parts). A workflow is given by id or as a Workflow object."""

    def __init__(self, project: Project2, rules: KeyRules | None = None, scan: InputScan | None = None):
        self.project = project
        self.rules = rules or key_rules()
        self.scan = scan if scan is not None else InputScan()
        self._pinned: dict[int, Workflow] = {}       # memos are per Workflow object: a what-if copy keeps its own
        self._chains: dict[int, list[str]] = {}
        self._keys: dict[tuple[int, str], str] = {}

    def _wf(self, wf: int | Workflow) -> Workflow:
        w = self.project.workflow(wf)
        self._pinned[id(w)] = w
        return w

    # ------------------------------------------------------------------ chains
    def chain(self, wf: int | Workflow) -> list[str]:
        """The stages of a workflow, in order."""
        w = self._wf(wf)
        if id(w) not in self._chains:
            self._chains[id(w)] = self.rules.chains.chain(w.options, w.end, reference=bool(self.project.reference_image),
                                                          learned_layer=wfl.has_learned_layer(w.params))
        return self._chains[id(w)]

    def deps(self, wf: int | Workflow, stage: str) -> tuple[str, ...]:
        """What a stage reads in the workflow's chain (deps_in_chain: optional and extra ones only when present)."""
        return self.rules.chains.deps_in_chain(self.chain(wf), stage)

    def ancestors(self, wf: int | Workflow, stage: str) -> list[str]:
        """Every stage the stage reads, directly or not, in the workflow's chain."""
        return self.rules.chains.ancestors(self.chain(wf), stage)

    # ------------------------------------------------------------------ the key
    def _given(self, w: Workflow, stage: str, exclude: list[str] | None) -> dict[str, Any]:
        """The values a stage gets over the automatic ones: stored < implied < injected (later wins). The excluded
        frames come from the definition only; a setting the stage no longer has (``retired``) is ignored."""
        gone = self.rules.retired(stage)
        v = {**{k: x for k, x in w.stored(stage).items() if k not in gone}, **self.rules.implied(w.options, stage)}
        if self.rules.injects_exclude(stage):
            v.pop("exclude", None)
            if exclude:
                v["exclude"] = exclude
        return v

    def _validated(self, w: Workflow, stage: str, values: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.rules.params(stage)(**values).model_dump(mode="json")
        except ValidationError as e:
            what = "; ".join(f"{stage}.{'.'.join(str(x) for x in er['loc'])}: {er['msg']}" for er in e.errors())
            raise ValueError(f"workflow {w.name!r}: invalid setting {what}") from None

    def key_params(self, wf: int | Workflow, stage: str) -> dict[str, Any]:
        """The parameter part of a stage's key (DESIGN §1.3)."""
        return self._key_values(self._wf(wf), stage)[0]

    def _key_values(self, w: Workflow, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """(the parameter part of the key, every validated value of the stage, defaults included)."""
        r = self.rules
        pairs = sorted(self.project.excluded) if r.injects_exclude(stage) else []
        v = self._given(w, stage, [name for _, name in pairs])       # validated as names, keyed as pairs
        v = {k: x for k, x in v.items() if k not in r.non_key(stage)}
        m = self._validated(w, stage, v)
        p = {k: m[k] for k in v}
        for k in r.path_fields(stage) & p.keys():
            if isinstance(p[k], str) and wfl.names_file(p[k]):
                p[k] = self.scan.file_id(_absolute(self.project, _split_tag(p[k])[0]))
        auto, default = r.auto_fields(stage), r.defaults(stage)
        p = {k: x for k, x in p.items() if k in auto or x != default.get(k)}
        for k, before in r.baseline(stage).items():      # a changed default: the effective value against the previous
            if k in m:
                if m[k] != before:
                    p[k] = m[k]
                else:
                    p.pop(k, None)
        p.update(r.constants(stage))
        if "exclude" in p and pairs:
            p["exclude"] = [list(t) for t in pairs]    # (session, name): the same after the raws move
        return p, m

    def project_part(self, stage: str) -> Any:
        """The project's part of a stage's key: the frame list for inventory, the target for astrometry, the image's
        file id for reference; None for the others."""
        p = self.project
        if stage == "inventory":
            return self.scan.frames(p)
        if stage == "astrometry":
            return {k: float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x
                    for k, x in asdict(p.target).items()}
        if stage == "reference":
            return self.scan.file_id(_absolute(p, p.reference_image)) if p.reference_image else None
        return None

    def key_parts(self, wf: int | Workflow, stage: str) -> dict[str, Any]:
        """What a stage's key hashes (the manifest keeps it as ``key_parts``)."""
        w = self._wf(wf)
        p, m = self._key_values(w, stage)
        return {"s": stage, "v": self.rules.version(stage, m), "p": p,
                "u": {d: self.key(w, d) for d in self.deps(w, stage)}, "x": self.project_part(stage)}

    def key(self, wf: int | Workflow, stage: str) -> str:
        """The key of a stage in a workflow (ValueError when the stage is not in its chain)."""
        w = self._wf(wf)
        k = self._keys.get((id(w), stage))
        if k is None:
            k = hashlib.sha1(canonical_json(self.key_parts(w, stage)).encode("ascii")).hexdigest()[:KEY_LEN]
            self._keys[(id(w), stage)] = k
        return k

    def keys(self, wf: int | Workflow) -> dict[str, str]:
        """{stage: key} for the workflow's chain."""
        return {s: self.key(wf, s) for s in self.chain(wf)}

    def sharing(self, wf: int | Workflow, stage: str) -> list[int]:
        """The ids of the project's workflows that have this stage with the same key (one result for all of them)."""
        k = self.key(wf, stage)
        return [o.id for o in self.project.workflows if stage in self.chain(o) and self.key(o, stage) == k]

    def block_key(self, wf: int | Workflow, block: str) -> tuple[str, ...]:
        """A block's key: the keys of its stages (Chains.key_stages); Output adds the workflow id (never shared)."""
        w = self._wf(wf)
        ks = tuple(self.key(w, s) for s in self.rules.chains.key_stages(block, self.chain(w)))
        return ks + ((f"workflow {w.id}",) if block == "output" else ())

    # ------------------------------------------------------------------ what a job gets
    def overrides(self, wf: int | Workflow, stage: str) -> dict[str, Any]:
        """The values a job of the stage runs with over the automatic ones: stored, implied and injected (the
        excluded frames' files), validated, paths absolute. Settings left out of the key are kept: they change how
        the job runs, not its result."""
        w = self._wf(wf)
        excl = self.scan.excluded_paths(self.project) if self.rules.injects_exclude(stage) else None
        v = self._given(w, stage, excl)
        m = self._validated(w, stage, v)
        out = {k: m[k] for k in v}
        for k in self.rules.path_fields(stage) & out.keys():
            if isinstance(out[k], str) and wfl.names_file(out[k]):
                path, tag = _split_tag(out[k])
                out[k] = _absolute(self.project, path) + tag
        return out


def stage_key(project: Project2, wf: int | Workflow, stage: str, scan: InputScan | None = None) -> str:
    """The key of one stage on the v2 stages (Keys(project).key)."""
    return Keys(project, scan=scan).key(wf, stage)


# ---------------------------------------------------------------------------------------------- folders

class Store:
    """The result folders of a v2 project: ``store/<stage>/<key>/`` a done result, ``<key>.tmp/`` a job in progress
    (never done), ``_trash/`` results replaced by a re-run, ``_absent/<stage>`` the folder of a stage a chain does
    not have (never created)."""

    def __init__(self, root: str | Path):
        self.path = Path(os.path.abspath(root)) / STORE
        self._usable: dict[tuple[str, str], bool] = {}

    @staticmethod
    def _check(key: str) -> str:
        if not isinstance(key, str) or not _KEY.match(key):
            raise ValueError(f"not a stage key: {key!r}")
        return key

    def dir(self, stage: str, key: str) -> Path:
        return self.path / stage / self._check(key)

    def tmp_dir(self, stage: str, key: str) -> Path:
        return self.path / stage / (self._check(key) + TMP)

    def absent_dir(self, stage: str) -> Path:
        return self.path / ABSENT / stage

    @property
    def trash(self) -> Path:
        return self.path / TRASH

    def manifest(self, stage: str, key: str) -> dict:
        """The manifest of a done folder ({} when there is none or it is unreadable)."""
        f = self.dir(stage, key) / "manifest.json"
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def done(self, stage: str, key: str) -> bool:
        """Whether ``store/<stage>/<key>/`` holds a done result of that key (a ``.tmp`` folder never does)."""
        m = self.manifest(stage, key)
        return m.get("status") == "done" and m.get("key", key) == key

    def usable_as_input(self, stage: str, key: str) -> bool:
        """Whether a done result can feed a job: false when it lost data outside its folder (inventory: a frame of
        frames.json; calibrate: the calibrated frame cache of cache.json). Cached per key: ``forget`` after a run."""
        if not self.done(stage, key):
            return False
        if (stage, key) not in self._usable:
            self._usable[(stage, key)] = _external_data_ok(stage, self.dir(stage, key))
        return self._usable[(stage, key)]

    def forget(self) -> None:
        self._usable.clear()


def _external_data_ok(stage: str, d: Path) -> bool:
    try:
        if stage == "inventory":
            frames = json.loads((d / "frames.json").read_text(encoding="utf-8"))
            return all(Path(f["path"]).is_file() for infos in frames.values() for f in infos)
        if stage == "calibrate":
            return Path(json.loads((d / "cache.json").read_text(encoding="utf-8"))["path"]).is_file()
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


# ---------------------------------------------------------------------------------------------- the shim

class StoreProject:
    """The project one job's stage code sees: the surface of core/project.Project that stages and v1's
    ``Pipeline._run_stage`` use, over the store (DESIGN §1.4).

    - ``stage_dir``: the job's own stage → ``<key>.tmp`` (or ``own_dir``: a resolve that must not touch the store);
      every ancestor → its done key folder; any other stage → ``store/_absent/<stage>``, never created.
    - ``config``: name, inputs (the union of the sessions), target, devices, reference image; no params or variants.
    - ``overrides(stage)``: stored + implied + injected values, validated. Every path handed out is absolute.
    - ``logs`` is the project's (one STOP file), ``results`` is export/, ``scratch`` is ``<scratch>/<key>``.
    """

    def __init__(self, keys: Keys, wf: int | Workflow, stage: str, *, own_dir: str | Path | None = None,
                 devices: dict[str, str] | None = None):
        p = keys.project
        if p.root is None:
            raise ValueError(f"project {p.name!r} has no folder yet")
        self.keys = keys
        self.workflow = p.workflow(wf)
        self.stage = stage
        self.chain = keys.chain(self.workflow)
        self.key = keys.key(self.workflow, stage)
        self.root = Path(os.path.abspath(p.root))
        self.store = Store(self.root)
        self.logs = self.root / "logs"
        self.results = self.root / "export"
        self.scratch = Path(os.path.abspath(p.scratch)) / self.key
        inputs = p.inputs()
        devs = p.devices if devices is None else devices
        self.config = ProjectConfig(
            name=p.name, inputs=Inputs(**{k: [_absolute(p, f) for f in getattr(inputs, k)] for k in KINDS}),
            target=copy.deepcopy(p.target), scratch_dir=str(self.scratch),
            reference_image=_absolute(p, p.reference_image) if p.reference_image else "",
            description=p.description, devices={k: v for k, v in devs.items() if v}, created=p.created)
        self.dirs: dict[str, Path] = {a: self.store.dir(a, keys.key(self.workflow, a))
                                      for a in keys.ancestors(self.workflow, stage)}
        self.dirs[stage] = Path(own_dir) if own_dir is not None else self.store.tmp_dir(stage, self.key)

    def stage_dir(self, stage: str) -> Path:
        d = self.dirs.get(stage)
        return d if d is not None else self.store.absent_dir(stage)

    def overrides(self, stage: str) -> dict[str, Any]:
        return self.keys.overrides(self.workflow, stage) if stage in self.chain else {}
