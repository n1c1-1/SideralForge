"""The store of a v2 project: stage results addressed by a key, and the project a job's stage code sees.

A done stage result lives in ``store/<stage>/<key>/``. Its key hashes everything that decides
the result, and nothing else::

    key(wf, s) = sha1(canonical_json({"s": s, "v": version, "p": key_params(wf, s),
                                      "u": {d: key(wf, d) for d in deps_in_chain(wf, s)},
                                      "x": project_part(s)}))[:16]

``key_params`` are the values the workflow stored, with those its options imply and the excluded frames on top,
validated through the stage's Params (2, 2.0 and "2" hash the same; an invalid value is a ValueError naming the
setting). The rule on defaults:

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
computed before it existed stays the same (tests/test_stacking.py).

Automatic values are not in the key: they follow from the upstream results, which the upstream keys stand for.
Settings that never change a result (worker counts) are left out. The key is the same in every process (canonical
JSON, no set or hash order).

The project parts read the disk (each session's frame files, the reference image): ``InputScan`` caches them. A scan
stats every frame: keep it out of the client's interface thread.

Issue #43: a chain entry may be an instance of a stage for one session, "calibrate@2". Its results live in the
stage's folder (``store/calibrate/<key>``); its key is that of a project holding that session alone: the stage's name,
its settings (the session's own, "calibrate@2" in the definition, over the stage's), the session's frames. A job of an
instance reads its session's instances under the stages' names; a job after the sessions' join reads the join as
"inventory" and "calibrate", and a workflow that starts from an image reads the image as "stack" (workflow.PROVIDES).

``StoreProject`` lets the stage code, and v1's ``Pipeline._run_stage``, run one job: its own stage writes into
``<key>.tmp`` (the runner renames it on success), every ancestor reads its done key folder, and any other stage
resolves to ``store/_absent/<stage>``, which is never created (``Context.has`` is false there).
"""
from __future__ import annotations

import copy
import hashlib
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
from .workflow import Chains, base, session_of

#: the target's fields added after the first keys were made: left out of the astrometry's key while unset (#38)
LATER_TARGET_FIELDS = ("pixel_um",)

STORE = "store"
TMP = ".tmp"                 # store/<stage>/<key>.tmp/: a job in progress, never done
TRASH = "_trash"             # done folders replaced by a re-run
ABSENT = "_absent"           # where the stages a chain does not have resolve; never created
KEY_LEN = 16
_KEY = re.compile(r"^[0-9a-f]{16}$")



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


#: an image source's content sample: this many pieces of PIECE bytes, spread over the file
PIECES, PIECE = 16, 4096


def content_sample(path: str | Path) -> str | None:
    """A short hash of pieces of a file spread over it (GUI exploration 7.7): two images of the same name, size and
    date in two folders (a batch export writes them in the same second) differ in it, a copy elsewhere keeps it.
    Reads 64 KB at most; None when the file cannot be read."""
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            h = hashlib.sha1()
            for i in range(PIECES):
                f.seek(max(0, (size - PIECE) * i // max(PIECES - 1, 1)))
                h.update(f.read(PIECE))
    except OSError:
        return None
    return h.hexdigest()[:16]


def _absolute(project: Project2, path: str | Path) -> str:
    """A path of the definition, absolute (relative to the project) and normalised, with '/'."""
    return Path(os.path.abspath(project.abspath(path))).as_posix()


# ---------------------------------------------------------------------------------------------- rules

class KeyRules:
    """What the key needs to know of the stages: their graph (``Chains``), the fields an ``auto()`` can set, the
    fields that never change a result (left out of the key; a job still gets them), the values workflow options imply,
    and the tables of changed defaults, removed settings and key constants.
    ``auto_fields`` None counts every field as automatic: nothing is ever dropped (the safe side)."""

    def __init__(self, chains: Chains, auto_fields: Mapping[str, Iterable[str]] | None = None,
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
        self._non_key = {s: frozenset(f) for s, f in (non_key or {}).items()}
        self.implied = implied
        self._defaults: dict[str, dict[str, Any]] = {}

    def params(self, stage: str) -> type[StageParams]:
        return self.chains.stages[wfl.base(stage)].Params     # a graph's later node ("denoise~2"): its stage's

    def version(self, stage: str, values: Mapping[str, Any] | None = None) -> str:
        """The stage's version in a key: its ``version``; with the definition's validated ``values``, the stage's
        ``key_version(values)`` when it has one (a part of the stage revised on its own, keyed only where the settings
        use it: the stack's Classic part)."""
        st = self.chains.stages[wfl.base(stage)]
        kv = getattr(st, "key_version", None)
        return str(kv(dict(values)) if callable(kv) and values is not None else st.version)

    def auto_fields(self, stage: str) -> frozenset[str]:
        """The fields an ``auto()`` can set (stages/v2.AUTO_FIELDS); without that table, every field (the safe
        side)."""
        if self._auto is None:
            return frozenset(self.params(stage).model_fields)
        return self._auto.get(wfl.base(stage), frozenset())

    def baseline(self, stage: str) -> dict[str, Any]:
        """{field: its previous default} of the fields whose default changed (stages/v2.KEY_BASELINE): the key holds
        their effective value, stored or not, unless it is that previous default."""
        return self._baseline.get(wfl.base(stage), {})

    def retired(self, stage: str) -> frozenset[str]:
        """The settings removed from the stage (stages/v2.RETIRED): a stored one is ignored."""
        return self._retired.get(wfl.base(stage), frozenset())

    def constants(self, stage: str) -> dict[str, Any]:
        """{field: value} of the removed settings whose last value stays in every key (stages/v2.KEY_CONSTANTS)."""
        return self._constants.get(wfl.base(stage), {})

    def non_key(self, stage: str) -> frozenset[str]:
        return self._non_key.get(wfl.base(stage), frozenset())

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
    """The rules of the v2 stages: ``workflow.chains()`` with the tables of stages/v2.py."""
    if not _RULES:
        from ..stages import v2
        _RULES.append(KeyRules(wfl.chains(), v2.AUTO_FIELDS, v2.NON_KEY, baseline=v2.KEY_BASELINE,
                               retired=v2.RETIRED, constants=v2.KEY_CONSTANTS))
    return _RULES[0]


# ---------------------------------------------------------------------------------------------- disk parts

def _useful_area(path: Path) -> list | None:
    """The useful area of a FITS frame as a key holds it; None for any other file, or a header that cannot be read."""
    from ..io.frames import fits_header, frame_kind, useful_area
    if frame_kind(path) != "fits":
        return None
    try:
        area = useful_area(fits_header(path))
    except Exception:  # noqa: BLE001 - a frame that cannot be read is said by the inventory, not by a key
        return None
    return list(area) if area is not None else None


class InputScan:
    """The parts of the keys that read the disk, cached: each session's frame files (by its folder and file lists)
    and file ids (the reference image). Call ``forget`` when files may have changed."""

    def __init__(self) -> None:
        self._sessions: dict[tuple, dict[str, list[tuple[Path, int]]]] = {}
        self._files: dict[str, list] = {}
        self._abs: dict[tuple[str, str], str] = {}
        self._frames: dict[tuple, list[list]] = {}     # frames(): the estimates of a large project ask it thousands
                                                        # of times (GUI exploration 4: 9 to 40 s at its opening)

    def forget(self) -> None:
        self._sessions.clear()
        self._files.clear()
        self._abs.clear()
        self._frames.clear()

    def _entries(self, project: Project2, session: Any) -> tuple:
        """A session's folder and file lists, absolute: what its frames are found from."""
        root = str(getattr(project, "root", ""))
        out = []
        for k in KINDS:
            row = []
            for f in getattr(session, k):
                a = self._abs.get((root, f))
                if a is None:
                    a = self._abs[(root, f)] = _absolute(project, f)
                row.append(a)
            out.append(tuple(row))
        return tuple(out)

    def session_frames(self, project: Project2, session: Any) -> dict[str, list[tuple[Path, int]]]:
        """{kind: [(file, size)]} of one session, as list_frames finds them."""
        entries = self._entries(project, session)
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

    def frames(self, project: Project2, sid: int | None = None, sessions: list | None = None) -> list[list]:
        """inventory's project part: [kind, file name, size] of each frame of the union of the sessions (``sid``: of
        that session only), per kind, in the order the inventory reads them (list_frames: by full path). Names and
        sizes, so copied raws keep the key; ordered, because the half-stack blocks follow the order.

        A frame of a session after the project's first carries that session's id (issue #56): capture programs
        number every night's frames alike ("Light_001.fits", the same size, even the same dates on a camera whose
        clock was reset), so that two nights' frames would be taken for the same ones. The first session's frames
        carry none: a project of one session, and the first night of several, keep their keys.

        The list is cached (until ``forget``) and shared: read it, never change it."""
        first = min((s.id for s in project.sessions), default=None)
        sessions = (sessions if sessions is not None else project.sessions) if sid is None else [project.session(sid)]
        memo = (first, tuple((s.id, self._entries(project, s)) for s in sessions))
        got = self._frames.get(memo)
        if got is None:
            got = self._frames[memo] = self._list_frames(project, sessions, first)
        return got

    def _list_frames(self, project: Project2, sessions: list, first: int | None) -> list[list]:
        out: list[list] = []
        for kind in KINDS:
            sizes: dict[Path, int] = {}
            tag: dict[Path, int | None] = {}
            for s in sessions:
                for f, n in self.session_frames(project, s)[kind]:
                    sizes.setdefault(f, n)
                    tag.setdefault(f, None if s.id == first else s.id)
            out += [[kind, f.name, sizes[f]] + ([tag[f]] if tag[f] is not None else []) for f in sorted(sizes)]
        return out

    def useful_areas(self, project: Project2, sessions: list) -> list[list]:
        """The FITS useful areas (io/frames.useful_area) the sessions' lights are read through, from the header of each
        session's first frame: [[session id, area]], empty for frames that have none (every frame of a camera raw file
        or of a FITS file without such a section, which then keeps its key). Results computed when the area was not
        read (full-size masters and calibrated frames) are not the same results."""
        out = []
        for s in sessions:
            lights = self.session_frames(project, s)["lights"]
            if not lights:
                continue
            k = "area:" + str(lights[0][0])
            if k not in self._files:
                self._files[k] = _useful_area(lights[0][0])
            if self._files[k] is not None:
                out.append([s.id, self._files[k]])
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

    def image_id(self, path: str) -> list:
        """An image source's id: its ``file_id`` and its ``content_sample`` (GUI exploration 7.7)."""
        k = "image:" + path
        if k not in self._files:
            fid = self.file_id(path)
            self._files[k] = fid + [content_sample(path) if fid[1] is not None else None]
        return self._files[k]


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
        if id(w) not in self._chains and getattr(w, "blocks", None) is not None:      # a graph (issue #47)
            self._chains[id(w)] = self.rules.chains.sequence(wfl.sequence_chain(
                w.blocks, wfl.source_of(w), reference=bool(self.project.reference_image),
                sessions=self.project.separate_sessions(w)))
        if id(w) not in self._chains:
            from .project2 import is_fusion, is_image, is_stretched
            src = getattr(w, "source", None)
            self._chains[id(w)] = self.rules.chains.chain(
                w.options, w.end, reference=bool(self.project.reference_image),
                sessions=self.project.separate_sessions(w), image=is_image(w) if src else False,
                mono=self.project.is_mono(w) if src is None else False, fusion=is_fusion(w) if src else False,
                stretched=is_stretched(w) if src else False)
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
        frames come from the definition only; a setting the stage no longer has (``retired``) is ignored. An
        instance gets the stage's stored values, then its own (issue #43: "calibrate@2" in the definition)."""
        b = base(stage)
        gone = self.rules.retired(b)
        stored = {**w.stored(b), **(w.stored(stage) if stage != b else {})}
        if "~" in stage:                                 # a later node of a graph's block: its own settings only
            stored = w.stored(stage)
        imp = wfl.node_implied(w.blocks, stage, wfl.source_of(w)) if getattr(w, "blocks", None) is not None \
            else self.rules.implied(w.options, b)
        v = {**{k: x for k, x in stored.items() if k not in gone}, **imp}
        if self.rules.injects_exclude(stage):
            v.pop("exclude", None)
            if exclude:
                v["exclude"] = exclude
        return v

    def _validated(self, w: Workflow, stage: str, values: dict[str, Any]) -> dict[str, Any]:
        stage = base(stage)
        try:
            return self.rules.params(stage)(**values).model_dump(mode="json")
        except ValidationError as e:
            what = "; ".join(f"{stage}.{'.'.join(str(x) for x in er['loc'])}: {er['msg']}" for er in e.errors())
            raise ValueError(f"workflow {w.name!r}: invalid setting {what}") from None

    def key_params(self, wf: int | Workflow, stage: str) -> dict[str, Any]:
        """The parameter part of a stage's key."""
        return self._key_values(self._wf(wf), stage)[0]

    def _key_values(self, w: Workflow, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """(the parameter part of the key, every validated value of the stage, defaults included)."""
        r = self.rules
        b = base(stage)
        pairs = sorted(self.project.excluded) if r.injects_exclude(b) else []
        v = self._given(w, stage, [name for _, name in pairs])       # validated as names, keyed as pairs
        v = {k: x for k, x in v.items() if k not in r.non_key(b)}
        m = self._validated(w, stage, v)
        p = {k: m[k] for k in v}
        auto, default = r.auto_fields(b), r.defaults(b)
        p = {k: x for k, x in p.items() if k in auto or x != default.get(k)}
        for k, before in r.baseline(b).items():      # a changed default: the effective value against the previous
            if k in m:
                if m[k] != before:
                    p[k] = m[k]
                else:
                    p.pop(k, None)
        p.update(r.constants(b))
        if "exclude" in p and pairs:
            p["exclude"] = [list(t) for t in pairs]    # (session, name): the same after the raws move
        return p, m


    def project_part(self, stage: str, w: Workflow | None = None) -> Any:
        """The project's part of a stage's key: the frame list for inventory, the target for astrometry, the image's
        file id for reference; None for the others."""
        p = self.project
        if base(stage) == "inventory":
            sid = session_of(stage)
            if sid is None and w is not None and w.sessions is not None:     # a filter's workflow: its sessions'
                frames, sessions = self.scan.frames(p, None, p.sessions_of(w)), p.sessions_of(w)
            else:
                frames = self.scan.frames(p, int(sid) if sid is not None else None)
                sessions = p.sessions if sid is None else [p.session(int(sid))]
            # frames whose header gives a useful area only: the key of every other project stays what it was
            areas = self.scan.useful_areas(p, sessions)
            return {"frames": frames, "useful_areas": areas} if areas else frames
        if stage == "astrometry":
            # a field added later (pixel_um, issue #38) counts only when it is set: the keys of the projects made
            # before it stay as they were
            return {k: float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x
                    for k, x in asdict(p.target).items() if not (k in LATER_TARGET_FIELDS and x is None)}
        if stage == "reference":
            return self.scan.file_id(_absolute(p, p.reference_image)) if p.reference_image else None
        return None

    def source_part(self, w: Workflow, stage: str) -> Any:
        """The source of a workflow that does not start from frames, for its first stage's key: an image file's id
        (name, size, date, a sample of its content; stretched or not); a fusion's palette and its inputs' keys (each
        input workflow's Output and Astrometry: a change in a channel's workflow makes the fusion Outdated)."""
        src = getattr(w, "source", None)
        if stage == "image" and src:
            # 7.7: with a sample of its content, so that two images alike by name, size and date differ
            fid = self.scan.image_id(_absolute(self.project, src["path"])) if src["path"] else ["", None, None]
            return fid + ["stretched"] if src.get("stretched") else fid
        if stage == "fusion" and src:
            f = src["fusion"]
            stars = f.get("stars")
            part = {"palette": f["palette"], "weights": dict(f.get("weights") or {}), "inputs": [
                {"role": i["role"], "channel": i.get("channel", "L"), "out": self._input_key(i["pid"], "render_simple"),
                 "wcs": self._input_key(i["pid"], "astrometry")}
                | ({"stars": self._input_key(i["pid"], "star_removal")} if stars else {})
                for i in f["inputs"]]}
            if stars:                                # option A: the starless channels, the star source
                part["star_source"] = dict(stars) | (
                    {"stars": self._input_key(stars["pid"], "star_removal"),
                     "wcs": self._input_key(stars["pid"], "astrometry")} if stars["mode"] == "workflow" else {})
            return part
        return None

    def _input_key(self, pid: int, stage: str) -> str | None:
        """The key of a stage of another workflow (a fusion's input), None when its chain has none."""
        try:
            return self.key(pid, stage) if stage in self.chain(pid) else None
        except (KeyError, ValueError):
            return None

    def fusion_inputs(self, wf: int | Workflow) -> list[tuple[int, str]]:
        """(workflow id, stage) of what a fusion reads: each input's Output and, when it has one, its Astrometry."""
        w = self._wf(wf)
        f = w.source["fusion"]
        want = ("render_simple", "astrometry") + (("star_removal",) if f.get("stars") else ())
        out = []
        for i in f["inputs"]:
            for st in want:
                if st in self.chain(i["pid"]) and (i["pid"], st) not in out:
                    out.append((i["pid"], st))
        stars = f.get("stars") or {}
        if stars.get("mode") == "workflow":
            for st in ("star_removal", "astrometry"):
                if st in self.chain(stars["pid"]) and (stars["pid"], st) not in out:
                    out.append((stars["pid"], st))
        return out

    def key_parts(self, wf: int | Workflow, stage: str) -> dict[str, Any]:
        """What a stage's key hashes (the manifest keeps it as ``key_parts``)."""
        w = self._wf(wf)
        p, m = self._key_values(w, stage)
        b = base(stage)
        x = self.source_part(w, stage) if b in ("image", "fusion") else self.project_part(stage, w)
        sid = session_of(stage)
        # an instance reads its session's instances under the stages' names: keyed so, as in a one-session project
        u = {(base(d) if sid is not None and session_of(d) == sid else d): self.key(w, d) for d in self.deps(w, stage)}
        # a block placed twice ("gradient~2") is keyed by its stage, not by its number: the same computation has the
        # same key whatever the graph numbered it (GUI exploration 2). Only "~" names change: no other key moves.
        u = {(k.split("~", 1)[0] if "~" in k and k.split("~", 1)[0] not in u else k): x for k, x in u.items()}
        return {"s": b, "v": self.rules.version(b, m), "p": p, "u": u, "x": x}

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
        excluded frames' files), validated. Settings left out of the key are kept: they change how the job runs, not
        its result."""
        w = self._wf(wf)
        excl = self.scan.excluded_paths(self.project) if self.rules.injects_exclude(base(stage)) else None
        v = self._given(w, stage, excl)
        if base(stage) == "image" and getattr(w, "source", None):
            v["path"] = _absolute(self.project, w.source["path"]) if w.source["path"] else ""
            v["stretched"] = bool(w.source.get("stretched"))
        if base(stage) == "fusion" and getattr(w, "source", None):
            store = Store(self.project.root) if self.project.root is not None else None
            v["palette"] = w.source["fusion"]["palette"]
            ins = []

            def folder(pid, st, file=""):
                k = self._input_key(pid, st)
                return str(store.dir(st, k) / file) if store and k else ""
            for i in w.source["fusion"]["inputs"]:
                ins.append({"role": i["role"], "channel": i.get("channel", "L"), "pid": i["pid"],
                            "dir": folder(i["pid"], "render_simple"), "stars_dir": folder(i["pid"], "star_removal"),
                            "wcs": folder(i["pid"], "astrometry", "wcs.fits")})
            v["inputs"] = ins
            stars = dict(w.source["fusion"].get("stars") or {})
            if stars.get("mode") == "workflow":
                stars.update(dir=folder(stars["pid"], "star_removal"), wcs=folder(stars["pid"], "astrometry", "wcs.fits"))
            v["stars"] = stars
            v["weights"] = dict(w.source["fusion"].get("weights") or {})
        m = self._validated(w, stage, v)
        return {k: m[k] for k in v}


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
        """``store/<stage>/<key>``; an instance's result lives in its stage's folder."""
        return self.path / base(stage) / self._check(key)

    def tmp_dir(self, stage: str, key: str) -> Path:
        return self.path / base(stage) / (self._check(key) + TMP)

    def absent_dir(self, stage: str) -> Path:
        return self.path / ABSENT / base(stage)

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
    stage = base(stage)
    try:
        if stage == "inventory":
            frames = json.loads((d / "frames.json").read_text(encoding="utf-8"))
            return all(Path(f["path"]).is_file() for infos in frames.values() for f in infos)
        if stage in ("calibrate", "sessions"):
            c = json.loads((d / "cache.json").read_text(encoding="utf-8"))
            return Path(c["path"]).is_file() and (not c.get("rgb_path") or Path(c["rgb_path"]).is_file())
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


# ---------------------------------------------------------------------------------------------- the shim

class StoreProject:
    """The project one job's stage code sees: the surface of core/project.Project that stages and v1's
    ``Pipeline._run_stage`` use, over the store.

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
        sid = session_of(stage)
        inputs = p.session_inputs(int(sid)) if sid is not None else p.inputs(self.workflow)
        if getattr(self.workflow, "source", None) is not None:
            inputs = Inputs()                            # an image source: no frames
        devs = p.devices if devices is None else devices
        self.config = ProjectConfig(
            name=p.name, inputs=Inputs(**{k: [_absolute(p, f) for f in getattr(inputs, k)] for k in KINDS}),
            target=copy.deepcopy(p.target), scratch_dir=str(self.scratch),
            reference_image=_absolute(p, p.reference_image) if p.reference_image else "",
            description=p.description, devices={k: v for k, v in devs.items() if v}, created=p.created)
        self.dirs: dict[str, Path] = {a: self.store.dir(a, keys.key(self.workflow, a))
                                      for a in keys.ancestors(self.workflow, stage)}
        self.dirs[stage] = Path(own_dir) if own_dir is not None else self.store.tmp_dir(stage, self.key)
        # the names the stage code reads (issue #43): an instance reads its session's instances under the stages'
        # names, and itself under its stage's; later stages read what stands for a stage (the join, the image)
        self.alias: dict[str, str] = {}
        for a in self.dirs:
            if session_of(a) is not None and session_of(a) == sid:
                self.alias[base(a)] = a
        for a in self.dirs:
            for d in wfl.PROVIDES.get(base(a), ()):
                self.alias.setdefault(d, a)

    def own_dir(self) -> Path:
        """The job's own folder (a graph's later node, "denoise~2", reads the first one as "denoise", issue #47)."""
        return self.dirs[self.stage]

    def stage_dir(self, stage: str) -> Path:
        d = self.dirs.get(stage)
        if d is None and stage in self.alias:
            d = self.dirs.get(self.alias[stage])
        return d if d is not None else self.store.absent_dir(stage)

    def overrides(self, stage: str) -> dict[str, Any]:
        if stage == base(self.stage):
            stage = self.stage                           # the job's own stage, an instance's settings included
        return self.keys.overrides(self.workflow, stage) if stage in self.chain else {}
