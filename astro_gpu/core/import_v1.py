"""Import of a v1 project into a v2 project, with the results that are still valid.

Nothing is ever written into the old project. The import and its dry run run under a write guard (an audit hook,
``sys.addaudithook``) that refuses any open-for-write, mkdir, rename, removal, copy or link into the old project's
folder, its scratch folder and its input files, and the import fails if a write was even attempted.

1. **The definition**: ``project2.from_v1`` (one session from the inputs, one workflow from main and one per
   variant), its input paths moved through ``path_map`` (an old folder or file → its new place or places). A
   reference image that no longer exists is dropped, with a note.
2. **Adoption.** Each workflow's chain is walked in order; a stage keeps its old result only if
   - its old manifest, read first, is done with the current stage version;
   - every stage it reads in the chain is adopted, and the old manifest's inputs name their old run ids;
   - its old settings equal a fresh resolve: v1's ``effective_params`` on a read-only view of the old results. The
     stage's own folder is a temporary one (``Context`` creates it), the stages it does not read are absent, the
     settings without effect on results (NON_KEY) are ignored;
   - inventory: the old frame list (its frames.json has no file size) and the new files agree on name, order,
     count, and each frame's EXIF capture time and exposure;
   - reference: the image is not newer than the old result.
3. **Copy**, never a link: each adopted result under its key, through ``store/<stage>/<key>.tmp`` renamed when
   complete. Only what later stages read is copied: outputs, previews and the manifest; never ``_ckpt``, temporary
   files or the calibrated frame cache (it lives in the scratch folder; calibrate's cache.json is pointed at the new
   scratch folder, so calibrate rebuilds the cache if register or stack must run again). frames.json paths are moved
   to the new raws, and the manifest gets ``key``, ``key_parts``, ``adopted_from`` and ``run_id = key``. The free
   space (the bytes to copy + 10 %) is checked before anything is written.
   The reference image goes to ``inputs/``.
4. **Report** (``ImportReport``): per workflow and stage, adopted (key) or recomputed (reason); the bytes to copy and
   the free space; notes; and the stage durations ``estimate.learn_from_v1`` reads in the old manifests (the import
   itself never writes a timing history).

``dry_run`` does 1, 2 and 4 and writes nothing at all (its temporary folder is a system one, removed at the end).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import estimate as est
from . import project2
from .files import write_json
from .journal import Journal
from .params import params_hash
from .project import Project, list_frames
from .project2 import KINDS, Project2, Workflow
from .runner import Pipeline, _same_params
from .store import InputScan, KeyRules, Keys, Store, StoreProject, file_id

REPORT_FILE = "import_v1.json"           # <new project>/logs/: the report of the import that made the project
SPACE_MARGIN = 0.10                      # the free space must exceed the bytes to copy by this share
SKIP_DIRS = frozenset({"_ckpt"})         # never copied: checkpoints of an interrupted run

PathMap = Mapping[str, "str | list[str]"]
Progress = Callable[[str], None]


class NotEnoughSpace(OSError):
    """The destination disk cannot hold the results to copy (+ 10 %)."""


# ---------------------------------------------------------------------------------------------- the write guard

class WriteRefused(PermissionError):
    """A write under a folder a ``WriteGuard`` protects."""


_ACTIVE: list[WriteGuard] = []
_HOOKED: list[bool] = []
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
# Windows CreateFile: GENERIC_WRITE, GENERIC_ALL, FILE_WRITE_DATA, FILE_APPEND_DATA, DELETE; and the dispositions that
# create or truncate (CREATE_NEW, CREATE_ALWAYS, OPEN_ALWAYS, TRUNCATE_EXISTING)
_WIN_WRITE = 0x40000000 | 0x10000000 | 0x0002 | 0x0004 | 0x00010000
_WIN_CREATE = (1, 2, 4, 5)
#: audit events and the arguments that are written to: the first, the destination, or both (a move, a link)
_FIRST = {"os.mkdir", "os.remove", "os.rmdir", "os.utime", "os.chmod", "os.chown", "os.truncate", "os.chflags",
          "os.lchflags", "os.setxattr", "os.removexattr", "shutil.rmtree", "shutil.chown"}
_DEST = {"shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.copytree", "_winapi.CopyFile2"}
_BOTH = {"os.rename", "os.link", "os.symlink", "shutil.move", "_winapi.CreateJunction"}


def _norm(path: Any) -> str | None:
    """A path as the guard compares it (absolute, normalised, case-folded on Windows); None for a file descriptor."""
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    if not isinstance(path, (str, os.PathLike)):
        return None
    try:
        return os.path.normcase(os.path.abspath(os.fspath(path)))
    except (TypeError, ValueError):
        return None


def _under(f: str, root: str) -> bool:
    return f == root or f.startswith(root.rstrip(os.sep) + os.sep)


def _audit(event: str, args: tuple) -> None:
    if not _ACTIVE:
        return
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                (isinstance(flags, int) and bool(flags & _WRITE_FLAGS)):
            _refuse(event, path)
    elif event in _FIRST:
        if args:
            _refuse(event, args[0])
    elif event in _DEST:
        if len(args) > 1:
            _refuse(event, args[1])
    elif event in _BOTH:
        for a in args[:2]:
            _refuse(event, a)
    elif event == "_winapi.CreateFile" and len(args) >= 4:
        access, disposition = args[1], args[3]
        if (isinstance(access, int) and access & _WIN_WRITE) or disposition in _WIN_CREATE:
            _refuse(event, args[0])


def _refuse(event: str, path: Any) -> None:
    f = _norm(path)
    if f is None:
        return
    for g in tuple(_ACTIVE):
        if g.covers_norm(f):
            msg = f"write guard: {event} {os.fspath(path) if not isinstance(path, bytes) else path!r} is under " \
                  f"a protected folder"
            g.refused.append(msg)
            raise WriteRefused(msg)


class WriteGuard:
    """While active (``with WriteGuard(folders):``), any write under the given folders or files raises
    ``WriteRefused``: an open for writing, mkdir, rename, removal, copy, link, time or mode change. Reads pass. Each
    refusal is recorded, and leaving the block raises if a refusal was swallowed by the code inside (an
    ``except OSError``), so an attempted write always fails the operation. The audit hook is installed once per
    process and costs nothing when no guard is active."""

    def __init__(self, folders: Iterable[str | Path] = ()):
        self.roots: list[str] = []
        self.refused: list[str] = []
        for f in folders:
            self.add(f)

    def add(self, folder: str | Path) -> None:
        r = _norm(folder) if str(folder).strip() else None
        if r and r not in self.roots:
            self.roots.append(r)

    def covers_norm(self, f: str) -> bool:
        return any(_under(f, r) for r in self.roots)

    def covers(self, path: str | Path) -> bool:
        f = _norm(path)
        return f is not None and self.covers_norm(f)

    def __enter__(self) -> WriteGuard:
        if not _HOOKED:
            sys.addaudithook(_audit)
            _HOOKED.append(True)
        _ACTIVE.append(self)
        return self

    def __exit__(self, et, e, tb) -> bool:
        _ACTIVE.remove(self)
        if et is None and self.refused:
            raise WriteRefused(f"{len(self.refused)} write(s) refused under a protected folder; first: "
                               f"{self.refused[0]}")
        return False


def snapshot(paths: Iterable[str | Path]) -> dict[str, tuple[bool, int, int]]:
    """Every folder and file under the given folders (and the given files): (is a folder, size, mtime in ns).
    Taken before and after an operation, equal snapshots prove that nothing was written, created or removed."""
    out: dict[str, tuple[bool, int, int]] = {}
    for root in paths:
        r = Path(root)
        if r.is_file():
            st = r.stat()
            out[str(r)] = (False, st.st_size, st.st_mtime_ns)
            continue
        for dp, dns, fns in os.walk(r):
            for n in ["", *dns, *fns]:
                f = Path(dp) / n if n else Path(dp)
                try:
                    st = f.stat()
                except OSError:
                    continue
                d = f.is_dir()
                out[str(f)] = (d, 0 if d else st.st_size, st.st_mtime_ns)
    return out


def snapshot_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """What differs between two snapshots: '+ path' (new), '- path' (gone), '~ path' (changed)."""
    out = [f"+ {k}" for k in after if k not in before] + [f"- {k}" for k in before if k not in after]
    return out + [f"~ {k}" for k in before if k in after and tuple(before[k]) != tuple(after[k])]


# ---------------------------------------------------------------------------------------------- paths

def _key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def map_path(path: str, path_map: PathMap | None) -> list[str]:
    """Where an input of the old project is now. The longest ``path_map`` entry that is the path or one of its
    folders gives the new place, the rest of the path kept (a value may list several folders: an old folder now
    split in two; a file then goes to the one that holds it). Unmapped paths are returned as they are."""
    if not path_map:
        return [path]
    k = _key(path)
    best: tuple[str, Any] | None = None
    for old, new in path_map.items():
        o = _key(old)
        if _under(k, o) and (best is None or len(o) > len(best[0])):
            best = (o, new)
    if best is None:
        return [path]
    o, new = best
    rest = os.path.normpath(str(path))[len(o):].lstrip("\\/")     # normcase keeps the length: the original case
    news = [new] if isinstance(new, (str, os.PathLike)) else list(new)
    if not rest:
        return [Path(n).as_posix() for n in news]
    cands = [Path(n) / rest for n in news]
    found = [c for c in cands if c.exists()]
    return [(found or cands)[0].as_posix()]


def _old_abs(old_root: Path, path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else old_root / p)


def folders_of(old_root: str | Path) -> list[str]:
    """The folders and files of a v1 project the import only reads: the project folder, its scratch folder, its
    inputs and its reference image (the write guard protects them)."""
    root = Path(os.path.abspath(old_root))
    d = json.loads((root / project2.V1_FILE).read_text(encoding="utf-8"))
    out = [str(root)]
    if d.get("scratch_dir"):
        out.append(_old_abs(root, d["scratch_dir"]))
    for kind in KINDS:
        out += [_old_abs(root, f) for f in (d.get("inputs") or {}).get(kind) or []]
    if d.get("reference_image"):
        out.append(_old_abs(root, d["reference_image"]))
    return list(dict.fromkeys(out))


def disk_free(path: str | Path) -> int:
    """Free bytes on the disk that holds ``path`` (or would hold it)."""
    p = Path(os.path.abspath(path))
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def result_files(folder: str | Path) -> list[tuple[Path, int]]:
    """(path relative to the folder, size) of what adoption copies from a result folder: its outputs, previews and
    manifest, without checkpoints (``_ckpt``) and temporary files. The calibrated frame cache is not there: it lives
    in the scratch folder."""
    folder = Path(folder)
    out = []
    for dp, dns, fns in os.walk(folder):
        dns[:] = sorted(n for n in dns if n not in SKIP_DIRS)
        for n in sorted(fns):
            if n.endswith(".tmp") or ".tmp." in n:
                continue
            f = Path(dp) / n
            out.append((f.relative_to(folder), f.stat().st_size))
    return out


def _read_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


# ---------------------------------------------------------------------------------------------- the report

@dataclass
class StageReport:
    """One stage of one workflow: adopted (its old result is copied under ``key``) or recomputed (``reason``)."""

    workflow: int
    stage: str
    key: str
    adopted: bool
    reason: str = ""
    source: str = ""            # the old result folder (adopted only)
    bytes: int = 0              # what copying it takes (adopted only)


@dataclass
class ImportReport:
    """What an import adopts and recomputes, per workflow and stage."""

    old_root: str
    new_root: str
    name: str = ""
    dry_run: bool = True
    workflows: dict[int, str] = field(default_factory=dict)
    stages: list[StageReport] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    copy_bytes: int = 0                                         # each adopted key once, plus inputs/
    free_bytes: int | None = None
    timings: list[list] = field(default_factory=list)          # [stage, seconds, units, kind] (estimate.learn_from_v1)
    written: bool = False
    project: Project2 | None = field(default=None, repr=False, compare=False)

    def of(self, wf: int) -> list[StageReport]:
        return [r for r in self.stages if r.workflow == wf]

    def adopted(self, wf: int | None = None) -> list[str]:
        """The adopted stages of a workflow (of the first one by default), in chain order."""
        wf = next(iter(self.workflows), None) if wf is None else wf
        return [r.stage for r in self.of(wf) if r.adopted]

    def recomputed(self, wf: int | None = None) -> list[tuple[str, str]]:
        """(stage, reason) of a workflow's stages that are recomputed (of the first workflow by default)."""
        wf = next(iter(self.workflows), None) if wf is None else wf
        return [(r.stage, r.reason) for r in self.of(wf) if not r.adopted]

    @property
    def enough_space(self) -> bool:
        return self.free_bytes is None or self.copy_bytes * (1 + SPACE_MARGIN) <= self.free_bytes

    def to_dict(self) -> dict[str, Any]:
        return {"old_root": self.old_root, "new_root": self.new_root, "name": self.name, "dry_run": self.dry_run,
                "written": self.written, "workflows": {str(k): v for k, v in self.workflows.items()},
                "stages": [asdict(r) for r in self.stages], "notes": list(self.notes),
                "copy_bytes": self.copy_bytes, "free_bytes": self.free_bytes, "timings": list(self.timings),
                "created": datetime.now().isoformat(timespec="seconds")}

    def text(self) -> str:
        """The report in a few lines (ASCII)."""
        how = "dry run" if self.dry_run else ("written" if self.written else "not written")
        lines = [f"{self.name}: {self.old_root} -> {self.new_root} ({how})"]
        for wid, wname in self.workflows.items():
            ad = self.adopted(wid)
            lines.append(f"  workflow {wid} '{wname}'")
            lines.append(f"    adopted ({len(ad)}): {', '.join(ad) or '-'}")
            rec = self.recomputed(wid)
            if rec:
                first, why = rec[0]
                rest = [s for s, _ in rec[1:]]
                lines.append(f"    recomputed ({len(rec)}): {first} ({why})" + (f", then {', '.join(rest)}"
                                                                                 if rest else ""))
        space = f"{_gb(self.copy_bytes)} to copy"
        if self.free_bytes is not None:
            space += f", {_gb(self.free_bytes)} free" + ("" if self.enough_space else " - NOT ENOUGH")
        lines.append(f"  {space}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- the import

def import_v1(old_root: str | Path, new_root: str | Path, path_map: PathMap | None = None, dry_run: bool = False,
              *, rules: KeyRules | None = None, progress: Progress | None = None) -> ImportReport:
    """Import the v1 project of ``old_root`` into a new v2 project at ``new_root``, adopting the results that are
    still valid (module docstring). ``path_map``: {old input folder or file: new place, or a list of places}.
    ``dry_run``: report only, write nothing. ``rules``: the stage rules (tests; default the v2 stages).

    Raises ValueError on a destination inside the old project's folders (or holding another project),
    NotEnoughSpace before any copy when the destination disk is too small, WriteRefused if anything tried to write
    into the old project."""
    old = Path(os.path.abspath(old_root))
    new = Path(os.path.abspath(new_root))
    if not (old / project2.V1_FILE).is_file():
        raise ValueError(f"not a v1 project (no {project2.V1_FILE}): {old}")
    say = progress or (lambda msg: None)
    with WriteGuard(folders_of(old)) as guard:
        if guard.covers(new) or _under(_norm(old), _norm(new)):
            raise ValueError(f"the new project {new} would hold or be inside the old project's folders")
        if not dry_run:
            _check_destination(old, new)
        with tempfile.TemporaryDirectory(prefix="astro_import_", ignore_cleanup_errors=True) as tmp:
            imp = _Import(old, new, path_map, rules, Path(tmp), say)
            for f in imp.sources():
                guard.add(f)
            rep = imp.check()
        rep.dry_run = dry_run
        rep.free_bytes = disk_free(new)
        try:
            rep.timings = [list(t) for t in est.learn_from_v1(old)]
        except Exception as e:  # noqa: BLE001 - durations are a hint
            rep.notes.append(f"stage durations not read: {e}")
        if not dry_run:
            if not rep.enough_space:
                raise NotEnoughSpace(f"not enough free space for the import into {new}: {_gb(rep.copy_bytes)} to copy "
                                     f"(+{SPACE_MARGIN:.0%}), {_gb(rep.free_bytes or 0)} free")
            imp.write(rep)
    return rep


def _check_destination(old: Path, new: Path) -> None:
    """A new folder, an empty one, or the folder an earlier import of the same project started (it resumes)."""
    if not new.exists():
        return
    if not new.is_dir():
        raise ValueError(f"not a folder: {new}")
    if not any(new.iterdir()):
        return
    rec = _read_json(new / "logs" / REPORT_FILE)
    if rec and _key(rec.get("old_root", "")) == _key(old):
        return
    raise ValueError(f"the folder of the new project is not empty: {new}")


class _Import:
    """One import: the definition, the adoption checks, then (``write``) the copy."""

    def __init__(self, old: Path, new: Path, path_map: PathMap | None, rules: KeyRules | None, tmp: Path,
                 say: Progress):
        self.old, self.new, self.tmp, self.say = old, new, tmp, say
        self.path_map = path_map
        self.v1 = Project(old)                                   # reads project.json only
        self.notes: list[str] = []
        self.defn = self._definition()
        self.keys = Keys(self.defn, rules, InputScan())
        self.chains = self.keys.rules.chains
        self.journal = Journal(tmp / "logs", echo=False)
        self.decisions: dict[tuple[str, str], str] = {}          # (key, old folder) → '' (adopt) or the reason
        self.adopt: dict[str, tuple[str, Path, Workflow]] = {}   # key → (stage, old folder, a workflow with it)
        self.keyset: dict[int, dict[str, str]] = {}
        self.frame_map: dict[str, str] = {}                      # old frame path (normalised) → new frame path

    # ------------------------------------------------------------------ the definition
    def _definition(self) -> Project2:
        p = project2.from_v1(self.old)
        for s in p.sessions:
            for kind in KINDS:
                out: list[str] = []
                for e in getattr(s, kind):
                    for n in map_path(e, self.path_map):
                        if n not in out:
                            out.append(n)
                setattr(s, kind, out)
        if p.reference_image:
            ref = map_path(_old_abs(self.old, p.reference_image), self.path_map)[0]
            if Path(ref).is_file():
                p.reference_image = Path(ref).as_posix()
            else:
                self.notes.append(f"the reference image {p.reference_image} no longer exists: dropped")
                p.reference_image = ""
        p.root = self.new
        return p

    def sources(self) -> list[str]:
        """The files the new definition reads (new raw places, reference): read only during the import."""
        p = self.defn
        out = [f for s in p.sessions for kind in KINDS for f in getattr(s, kind)]
        if p.reference_image:
            out.append(p.reference_image)
        return [f for f in out if not _under(_norm(f) or "", _norm(self.new))]

    def _variant(self, w: Workflow) -> str:
        return w.name if w.variant_of is not None and w.name in self.v1.config.variants else ""

    def _old_dir(self, w: Workflow, stage: str) -> Path:
        cfg = self.v1.config
        cfg.active_variant = self._variant(w)                    # in memory only: project.json is never saved
        try:
            return Path(self.v1.stage_dir(stage))
        finally:
            cfg.active_variant = ""

    # ------------------------------------------------------------------ the checks
    def check(self) -> ImportReport:
        p = self.defn
        rep = ImportReport(str(self.old), str(self.new), p.name, workflows={w.id: w.name for w in p.workflows})
        sizes: dict[str, int] = {}
        for w in p.workflows:
            try:
                chain = self.keys.chain(w)
                self.keyset[w.id] = self.keys.keys(w)
            except ValueError as e:
                rep.notes.append(f"workflow '{w.name}' cannot be keyed: {e}")
                continue
            dirs = {s: self._old_dir(w, s) for s in chain}
            adopted: dict[str, bool] = {}
            self.say(f"{p.name}: checking workflow '{w.name}'")
            for s in chain:
                k = self.keyset[w.id][s]
                memo = (k, str(dirs[s]))
                if memo not in self.decisions:
                    self.decisions[memo] = self._decide(w, s, chain, dirs, adopted)
                why = self.decisions[memo]
                adopted[s] = not why
                size = 0
                if not why:
                    if k not in sizes:
                        sizes[k] = sum(n for _, n in result_files(dirs[s]))
                    size = sizes[k]
                    self.adopt.setdefault(k, (s, dirs[s], w))
                rep.stages.append(StageReport(w.id, s, k, not why, why, "" if why else str(dirs[s]), size))
        rep.copy_bytes = sum(sizes.values()) + sum(Path(f).stat().st_size for f in self._inputs_to_copy())
        rep.notes = self.notes + rep.notes
        rep.project = p
        return rep

    def _manifest(self, d: Path) -> dict:
        return _read_json(d / "manifest.json")

    def _decide(self, w: Workflow, s: str, chain: list[str], dirs: dict[str, Path], adopted: dict[str, bool]) -> str:
        """'' when the old result of ``s`` is adopted, else why it is recomputed."""
        m = self._manifest(dirs[s])
        stage = self.chains.stages[s]
        if not m:
            return "no result in the old project"
        if m.get("status") != "done":
            return f"the old result is {m.get('status') or 'incomplete'}"
        if str(m.get("version")) != str(stage.version):
            return f"algorithm updated ({m.get('version')} -> {stage.version})"
        for d in self.keys.deps(w, s):
            if not adopted.get(d):
                return f"'{d}' is recomputed"
        inputs = m.get("inputs") or {}
        for d in stage.deps:
            if d in chain and inputs.get(d) != self._manifest(dirs[d]).get("run_id"):
                return f"'{d}' was run again after this result"
        why = self._params(w, s, m, dirs)
        if why:
            return why
        if s == "inventory":
            return self._frames(m, dirs[s])
        if s == "reference":
            return self._reference(m)
        return ""

    def _params(self, w: Workflow, s: str, m: dict, dirs: dict[str, Path]) -> str:
        """'' when the old settings equal a fresh resolve (v1's effective_params on the old results)."""
        stage = self.chains.stages[s]
        try:
            sp = StoreProject(self.keys, w, s, own_dir=self.tmp / f"wf{w.id}" / s)
            for a in list(sp.dirs):
                if a != s:
                    sp.dirs[a] = dirs[a]                         # ancestors: the old results, read only
            p, _, _ = Pipeline(sp, [stage], self.journal).effective_params(stage)
        except WriteRefused:
            raise
        except Exception as e:  # noqa: BLE001 - any failure means: not adopted
            return f"the settings could not be checked: {type(e).__name__}: {e}"
        cur = p.model_dump(mode="json")
        stored = dict(m.get("params") or {})
        for k in self.keys.rules.non_key(s):
            if k in cur:
                stored[k] = cur[k]
        if params_hash(p) == m.get("params_hash") or _same_params(stage, p, stored):
            return ""
        diff = [k for k in cur if k in stored and stored[k] != cur[k]] or [k for k in cur if k not in stored]
        return "settings changed: " + ", ".join(f"{k} {stored.get(k, 'unset')!r} -> {cur[k]!r}" for k in diff[:3])

    def _frames(self, m: dict, d: Path) -> str:
        """'' when the old frame list and the new files agree on name, order, count, EXIF capture time and
        exposure (the old frames.json has no file size)."""
        try:
            old = json.loads((d / "frames.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "the old frame list (frames.json) cannot be read"
        inputs = self.defn.inputs()
        max_lights = int((m.get("params") or {}).get("max_lights") or 0)
        pairs: list[tuple[str, Path, dict]] = []
        for kind in KINDS:
            files = list_frames(getattr(inputs, kind))
            if kind == "lights" and max_lights and max_lights < len(files):     # as the inventory picks them
                keep = np.unique(np.linspace(0, len(files) - 1, max_lights).round().astype(int))
                files = [files[i] for i in keep]
            was = [e for e in old.get(kind) or [] if isinstance(e, dict)]
            if len(files) != len(was):
                return f"{kind}: {len(files)} files now, {len(was)} in the old result"
            for i, (f, e) in enumerate(zip(files, was)):
                name = Path(str(e.get("path", ""))).name
                if os.path.normcase(f.name) != os.path.normcase(name):
                    return f"{kind} #{i + 1}: {f.name} now, {name} in the old result (other files or another order)"
            pairs += [(kind, f, e) for f, e in zip(files, was)]
        self.say(f"reading the capture time and exposure of {len(pairs)} frames")
        with ThreadPoolExecutor(8) as ex:
            infos = list(ex.map(_frame_info, pairs))
        for (kind, f, e), info in zip(pairs, infos):
            if isinstance(info, Exception):
                return f"{kind} {f.name}: cannot be read ({info})"
            if info.time != e.get("time"):
                return f"{kind} {f.name}: captured {info.time} now, {e.get('time')} in the old result"
            if not _same_number(info.exposure, e.get("exposure")):
                return f"{kind} {f.name}: exposure {info.exposure} s now, {e.get('exposure')} s in the old result"
        self.frame_map.update({_key(e["path"]): str(f) for _, f, e in pairs})
        return ""

    def _reference(self, m: dict) -> str:
        """'' when the reference image is not newer than the old result (it was made from this image)."""
        try:
            started = datetime.fromisoformat(m.get("started") or m.get("finished") or "").timestamp()
            changed = Path(self.defn.reference_path()).stat().st_mtime
        except (OSError, ValueError, TypeError):
            return "the date of the old result or of the image is unknown"
        return "the reference image is newer than the old result" if changed > started + 2.0 else ""

    def _inputs_to_copy(self) -> list[str]:
        """The files that go to inputs/: the reference image."""
        p = self.defn
        return [p.reference_image] if p.reference_image and Path(p.reference_image).is_file() else []

    # ------------------------------------------------------------------ the copy
    def write(self, rep: ImportReport) -> None:
        root = self.new
        (root / "logs").mkdir(parents=True, exist_ok=True)
        write_json(root / "logs" / REPORT_FILE, {**rep.to_dict(), "status": "copying"})
        self._move_inputs()
        again = Keys(self.defn, self.keys.rules, InputScan())
        for w in self.defn.workflows:
            if w.id in self.keyset and again.keys(w) != self.keyset[w.id]:
                raise RuntimeError(f"workflow '{w.name}': the keys changed when the inputs moved to inputs/")
        project2.save(self.defn, root)
        store = Store(root)
        for i, (k, (s, src, w)) in enumerate(self.adopt.items()):
            if store.done(s, k):
                continue                                         # copied by an earlier, interrupted import
            self.say(f"copying {s} ({i + 1}/{len(self.adopt)})")
            tmp = store.tmp_dir(s, k)
            if tmp.exists():
                shutil.rmtree(tmp)                               # this import's own unfinished copy
            for rel, _ in result_files(src):
                dst = tmp / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / rel, dst)
            self._rewrite(s, k, w, src, tmp)
            os.replace(tmp, store.dir(s, k))
        rep.written = True
        rep.project = self.defn
        write_json(root / "logs" / REPORT_FILE, {**rep.to_dict(), "status": "done"})

    def _move_inputs(self) -> None:
        """Copy the reference image into inputs/ (same name and date, so the same keys) and point the definition at
        it."""
        p = self.defn
        p.inputs_dir.mkdir(parents=True, exist_ok=True)
        placed: dict[str, str] = {}                              # source (normalised) → path in the project
        for f in self._inputs_to_copy():
            name = Path(f).name
            rel, i = Path("inputs") / name, 2
            while rel.as_posix() in placed.values() or \
                    ((p.root / rel).exists() and file_id(p.root / rel) != file_id(f)):
                rel, i = Path("inputs") / str(i) / name, i + 1   # another file of that name: its own folder, same name
            (p.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, p.root / rel)
            placed[_norm(f) or f] = rel.as_posix()
        if p.reference_image:
            p.reference_image = placed.get(_norm(p.reference_image) or "", p.reference_image)

    def _rewrite(self, s: str, k: str, w: Workflow, src: Path, tmp: Path) -> None:
        """The adopted copy as a v2 result: its manifest with the key fields, frames.json on the new raws,
        calibrate's cache.json on the new scratch folder."""
        m = self._manifest(src)
        keys = self.keyset[w.id]
        m["inputs"] = {d: keys[d] for d in (m.get("inputs") or {}) if d in keys}
        m.update(run_id=k, key=k, key_parts=self.keys.key_parts(w, s), run=None,
                 adopted_from={"project": str(self.old), "folder": str(src), "run_id": m.get("run_id"),
                               "finished": m.get("finished"), "imported": datetime.now().isoformat(timespec="seconds")})
        for f in tmp.rglob("frames.json"):
            write_json(f, _relocate_frames(json.loads(f.read_text(encoding="utf-8")), self.frame_map))
        cache = tmp / "cache.json"
        if s == "calibrate" and cache.is_file():
            c = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(c, dict) and c.get("path"):
                c["path"] = str(Path(os.path.abspath(self.defn.scratch)) / k / s / Path(str(c["path"])).name)
                write_json(cache, c)
        write_json(tmp / "manifest.json", m)


def _relocate_frames(obj: Any, frame_map: dict[str, str]) -> Any:
    """frames.json with each frame's "path" moved to the new raws."""
    if isinstance(obj, dict):
        return {k: (frame_map.get(_key(v), v) if k == "path" and isinstance(v, str) else _relocate_frames(v, frame_map))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relocate_frames(x, frame_map) for x in obj]
    return obj


def _frame_info(pair: tuple[str, Path, dict]) -> Any:
    """The capture time and exposure of a new frame (only the EXIF header of a RAW file is read: its sensor
    properties come from the old entry), or the exception that prevented it."""
    from ..io.frames import FrameInfo, read_info
    _, f, e = pair
    try:
        like = FrameInfo(**{k: v for k, v in e.items() if k in {x.name for x in fields(FrameInfo)}})
    except TypeError:                        # an old entry without the sensor properties: the file is decoded
        like = None
    try:
        return read_info(f, like=like if like is not None and like.kind == "raw" else None)
    except Exception as ex:  # noqa: BLE001 - reported as a mismatch
        return ex


def _same_number(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a == b
