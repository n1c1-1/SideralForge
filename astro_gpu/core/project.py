"""A v1 processing project (project.json): one target, its input folders and everything computed from them.

v1 projects run through ``core/runner.Pipeline`` (the ``astro-gpu`` command line); the client opens them through the
import (``core/import_v1.py``). The v2 code (``core/project2.py``, ``core/store.py``...) reuses ``Inputs``,
``Target``, ``ProjectConfig``, ``list_frames`` and ``frame_kind`` from here.

Layout on disk::

    <project>/
        project.json        inputs, target, user parameter overrides, render settings
        work/<stage>/       outputs of each stage + manifest.json (+ _ckpt/ while a stage can be resumed)
        results/            exported images
        variants/<slug>/<stage>/  the stages a variant computes its own way (see below)
        logs/events.jsonl   machine-readable journal (progress, metrics, previews)
        logs/journal.md     human-readable log

Large temporary data (the calibrated frames cache) lives in ``scratch_dir``, ideally on a fast local SSD.

Variants: the same data processed another way, kept side by side so that methods can be compared. A variant is a
set of parameter changes on top of the project's own ('main'); the stages those changes touch, and every stage
after them, get their own folders under ``variants/<slug>/``; the stages before are shared with main. The active
variant (``active_variant``, '' = main) is the one the runs, the rendering and the exports use.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

FORMAT = 1
RAW_EXT = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".orf", ".rw2", ".pef", ".srw", ".raf"}
FITS_EXT = {".fit", ".fits", ".fts"}
FITS_PACKED = {".fz", ".gz"}         # compressed FITS: .fits.fz (tile-compressed), .fit.gz (gzip)
COLOUR_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}      # frames already in colour (issue #57)


def frame_kind(path: str | Path) -> str:
    """'raw', 'fits' (compressed FITS included), 'colour' (a JPEG, PNG or TIFF: a frame already in colour, issue #57)
    or '' for a file that is not a frame."""
    name = Path(path).name.lower()
    ext = Path(name).suffix
    if ext in RAW_EXT:
        return "raw"
    if ext in FITS_EXT or ext == ".fz" or (ext in FITS_PACKED and Path(name[:-len(ext)]).suffix in FITS_EXT):
        return "fits"
    return "colour" if ext in COLOUR_EXT else ""


def file_type(path: str | Path) -> str:
    """What a file of a session is, as the user names it: its camera raw format ("ARW", "CR2"...), "FITS" (every FITS
    form), or its extension for any other file ("JPG")."""
    kind = frame_kind(path)
    if kind == "fits":
        return "FITS"
    return Path(path).suffix.lstrip(".").upper() or "?"


def type_warnings(files: dict[str, list], masters: set | frozenset = frozenset()) -> list[dict]:
    """What may be a mistake in one session's files ({kind: [path]}), issue #58: a file the engine does not read (not a
    frame; or a JPEG, PNG or TIFF among raw or FITS frames, which leaves it aside); lights of mixed types (1 JPG among
    14 ARW); calibration frames of another type than the lights, unless they are masters (``masters``: the kinds given
    as masters only, io/frames.master_kinds: a master FITS file beside raw lights is the normal case, and its scale is
    read as the lights'); and calibration frames that cannot calibrate these lights (issue #57: frames already in
    colour against mosaics, one way or the other); the same files given as two kinds (said on the later kind,
    ``other`` the earlier one).
    Each: {"key": "unread" | "mixed" | "other_type" | "wrong_kind" | "same_files", "kind", "type", "n", "main", "of",
    "name", "other"}."""
    out: list[dict] = []
    main: dict[str, str] = {}
    for kind, paths in files.items():
        if not paths:
            continue
        types: dict[str, list] = {}
        for f in paths:
            types.setdefault(file_type(f), []).append(f)
        top = max(types, key=lambda t: (len(types[t]), bool(frame_kind(types[t][0]))))
        main[kind] = top
        colour = frame_kind(types[top][0]) == "colour"
        for t, fs in sorted(types.items()):
            k = frame_kind(fs[0])
            if not k or (k == "colour" and not colour):
                out.append({"key": "unread", "kind": kind, "type": t, "n": len(fs), "name": Path(fs[0]).name})
            elif t != top:
                out.append({"key": "mixed", "kind": kind, "type": t, "n": len(fs), "main": top, "of": len(types[top]),
                            "name": Path(fs[0]).name})
    light = main.get("lights")
    for kind, t in main.items():
        a, b = frame_kind("x." + t.lower()), frame_kind("x." + light.lower()) if light else ""
        if kind != "lights" and t != light and a and b:
            key = "wrong_kind" if (a == "colour") != (b == "colour") else "other_type"
            if key == "other_type" and kind in masters:
                continue
            out.append({"key": key, "kind": kind, "type": t, "main": light, "n": len(files[kind])})
    kinds = [k for k, paths in files.items() if paths]
    names = {k: {os.path.normcase(str(f)) for f in files[k]} for k in kinds}
    for i, kind in enumerate(kinds):
        for other in kinds[:i]:
            if names[kind] & names[other]:
                out.append({"key": "same_files", "kind": kind, "other": other, "n": len(names[kind] & names[other])})
    return out


def colour_lights(files: list) -> bool:
    """Whether a list of light frames holds frames already in colour by their names: most of them are JPEG, PNG or
    TIFF (the others are said by ``type_warnings``). An RGB FITS frame is found from its header (io/frames)."""
    n = sum(1 for f in files if frame_kind(f) == "colour")
    return n > 0 and 2 * n >= len(files)


@dataclass
class Inputs:
    lights: list[str] = field(default_factory=list)
    darks: list[str] = field(default_factory=list)
    flats: list[str] = field(default_factory=list)
    bias: list[str] = field(default_factory=list)
    flat_darks: list[str] = field(default_factory=list)


@dataclass
class Target:
    name: str = ""
    ra: float | None = None          # degrees, optional hint for astrometry
    dec: float | None = None
    focal_mm: float | None = None    # effective focal length, optional (makes the astrometry immediate)
    pixel_um: float | None = None    # the camera's pixel size, optional (with the focal length: the image scale)


@dataclass
class ProjectConfig:
    name: str
    inputs: Inputs = field(default_factory=Inputs)
    target: Target = field(default_factory=Target)
    scratch_dir: str = ""
    reference_image: str = ""        # optional: a previous processing of the same data, for comparison
    description: str = ""            # free text: instrument, frames, what was done, what to look at
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    render: dict[str, Any] = field(default_factory=dict)
    devices: dict[str, str] = field(default_factory=dict)
    variants: dict[str, dict[str, Any]] = field(default_factory=dict)   # name -> {"params", "note", "created", "slug"}
    active_variant: str = ""                                            # '' = main
    created: str = ""
    format: int = FORMAT


class Project:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.config = self._load()

    # ------------------------------------------------------------------ creation / persistence
    @classmethod
    def create(cls, root: str | Path, name: str, inputs: Inputs, target: Target | None = None,
               scratch_dir: str | Path | None = None, reference_image: str = "") -> Project:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        cfg = ProjectConfig(name=name, inputs=inputs, target=target or Target(),
                            scratch_dir=str(scratch_dir or root / "work" / "_scratch"),
                            reference_image=reference_image, created=datetime.now().isoformat(timespec="seconds"))
        (root / "project.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        p = cls(root)
        remember(p)
        return p

    def _load(self) -> ProjectConfig:
        p = self.root / "project.json"
        if not p.exists():
            raise FileNotFoundError(f"not a project folder (no project.json): {self.root}")
        d = json.loads(p.read_text(encoding="utf-8"))
        return ProjectConfig(name=d["name"], inputs=Inputs(**d.get("inputs", {})), target=Target(**d.get("target", {})),
                             scratch_dir=d.get("scratch_dir", ""), reference_image=d.get("reference_image", ""),
                             description=d.get("description", ""), params=d.get("params", {}), render=d.get("render", {}),
                             devices=d.get("devices", {}),
                             variants=d.get("variants", {}), active_variant=d.get("active_variant", ""),
                             created=d.get("created", ""), format=d.get("format", FORMAT))

    def save(self) -> None:
        tmp = self.root / "project.json.tmp"
        tmp.write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")
        os.replace(tmp, self.root / "project.json")

    # ------------------------------------------------------------------ paths
    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def scratch(self) -> Path:
        return Path(self.config.scratch_dir) if self.config.scratch_dir else self.work / "_scratch"

    def stage_dir(self, stage: str) -> Path:
        v = self.config.active_variant
        if v and v in self.config.variants and stage in self.variant_stages(v):
            return self.variant_dir(v) / stage
        return self.work / stage

    # ------------------------------------------------------------------ parameters
    def overrides(self, stage: str) -> dict[str, Any]:
        """The user's parameter values of a stage: main's, with the active variant's changes on top (a change to
        None puts the parameter back to its automatic / default value)."""
        out = dict(self.config.params.get(stage, {}))
        v = self.config.active_variant
        if v and v in self.config.variants:
            for k, val in self.config.variants[v].get("params", {}).get(stage, {}).items():
                if val is None:
                    out.pop(k, None)
                else:
                    out[k] = val
        return out

    def set_override(self, stage: str, name: str, value: Any) -> None:
        """Set a parameter of the active variant, or of main when no variant is active."""
        v = self.config.active_variant
        if v and v in self.config.variants:
            self.config.variants[v].setdefault("params", {}).setdefault(stage, {})[name] = value
        else:
            self.config.params.setdefault(stage, {})[name] = value
        self.save()

    def clear_override(self, stage: str, name: str) -> None:
        v = self.config.active_variant
        if v and v in self.config.variants:
            vp = self.config.variants[v].setdefault("params", {}).setdefault(stage, {})
            if name in self.config.params.get(stage, {}):
                vp[name] = None                  # main sets it: the variant goes back to the automatic value
            else:
                vp.pop(name, None)
                if not vp:
                    self.config.variants[v]["params"].pop(stage, None)
        else:
            self.config.params.get(stage, {}).pop(name, None)
        self.save()

    # ------------------------------------------------------------------ variants
    def variant_dir(self, name: str) -> Path:
        return self.root / "variants" / self.config.variants[name]["slug"]

    def variant_stages(self, name: str) -> set[str]:
        """The stages a variant computes its own way: those whose parameters it changes, and every stage after them
        (they read what those produce). The others are shared with main."""
        changed = {st for st, kv in self.config.variants.get(name, {}).get("params", {}).items() if kv}
        if not changed:
            return set()
        graph = stage_graph()
        out = set(changed)
        grew = True
        while grew:
            grew = False
            for st, deps in graph.items():
                if st not in out and any(d in out for d in deps):
                    out.add(st)
                    grew = True
        return out

    def create_variant(self, name: str, params: dict[str, dict[str, Any]], note: str = "", prune: bool = True) -> None:
        """A new variant (not active): ``params`` = {stage: {parameter: value}} changed from main (None = back to
        the automatic value). ``prune``: drop the changes to automatic of parameters main leaves automatic (no
        change, and they would give the variant stages of its own for nothing)."""
        if prune:
            params = {st: {k: v for k, v in kv.items() if not (v is None and k not in self.config.params.get(st, {}))}
                      for st, kv in params.items()}
        name = name.strip()
        if not name or name.lower() == "main" or name in self.config.variants:
            raise ValueError(f"a variant needs a new name (not 'main'): {name!r}")
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:40] or "variant"
        taken = {v.get("slug") for v in self.config.variants.values()}
        base, k = slug, 2
        while slug in taken or (self.root / "variants" / slug).exists():
            slug, k = f"{base}_{k}", k + 1
        self.config.variants[name] = {"params": {st: dict(kv) for st, kv in params.items() if kv}, "note": note,
                                      "created": datetime.now().isoformat(timespec="seconds"), "slug": slug}
        self.save()

    def set_active_variant(self, name: str) -> None:
        if name and name not in self.config.variants:
            raise KeyError(f"no variant named {name!r}")
        self.config.active_variant = name
        self.save()

    def delete_variant(self, name: str) -> None:
        """Remove a variant and everything it computed (main is never touched)."""
        if name not in self.config.variants:
            return
        d = self.variant_dir(name)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        del self.config.variants[name]
        if self.config.active_variant == name:
            self.config.active_variant = ""
        self.save()

    def promote_variant(self, name: str, keep_main_as: str = "") -> None:
        """Make a variant the project's main processing: its parameter changes go into main and its stage folders
        replace main's. Main's former folders for those stages are kept as a variant named ``keep_main_as``, or
        deleted when that name is empty."""
        if name not in self.config.variants:
            raise KeyError(f"no variant named {name!r}")
        own = self.variant_stages(name)
        vparams = self.config.variants[name].get("params", {})
        vdir = self.variant_dir(name)
        kdir = None
        if keep_main_as:
            old = {st: {k: self.config.params.get(st, {}).get(k) for k in kv} for st, kv in vparams.items()}
            self.create_variant(keep_main_as, old, note=f"main before '{name}' became main", prune=False)
            kdir = self.variant_dir(keep_main_as)
            kdir.mkdir(parents=True, exist_ok=True)
        for st in own:
            src, dst = vdir / st, self.work / st
            if dst.exists():
                if kdir is not None:
                    shutil.move(str(dst), str(kdir / st))
                else:
                    shutil.rmtree(dst, ignore_errors=True)
            if src.exists():
                shutil.move(str(src), str(dst))
        for st, kv in vparams.items():
            for k, val in kv.items():
                if val is None:
                    self.config.params.get(st, {}).pop(k, None)
                else:
                    self.config.params.setdefault(st, {})[k] = val
        shutil.rmtree(vdir, ignore_errors=True)
        del self.config.variants[name]
        if self.config.active_variant == name:
            self.config.active_variant = ""
        self.save()

    def variant_size(self, name: str) -> int:
        """Bytes on disk of what a variant computed."""
        d = self.variant_dir(name)
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0


_GRAPH: dict[str, tuple[str, ...]] = {}


def stage_graph() -> dict[str, tuple[str, ...]]:
    """{stage: its dependencies}, in pipeline order."""
    if not _GRAPH:
        from ..stages import all_stages
        _GRAPH.update({s.name: tuple(s.deps) for s in all_stages()})
    return _GRAPH


#: per-user state: the list of recent projects, the stage durations, the camera colour profiles. ASTRO_GPU_HOME moves
#: it elsewhere, so that test runs (tools/qc) neither appear among the user's projects nor teach the user's camera
#: profiles (the download cache stays in ~/.astro_gpu/cache, shared)
STATE = Path(os.environ.get("ASTRO_GPU_HOME") or Path.home() / ".astro_gpu")
REGISTRY = STATE / "projects.json"


def remember(project: Project) -> None:
    """Record a project as recently used (shared by the command line and the interface)."""
    try:
        reg = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {}
    except (OSError, ValueError):
        reg = {}
    key = str(project.root.resolve())
    reg[key] = {"name": project.config.name, "used": datetime.now().isoformat(timespec="seconds"),
                "description": project.config.description}
    reg = dict(sorted(reg.items(), key=lambda kv: kv[1]["used"], reverse=True)[:30])
    try:
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        tmp = REGISTRY.with_suffix(".tmp")
        tmp.write_text(json.dumps(reg, indent=1), encoding="utf-8")
        os.replace(tmp, REGISTRY)
    except OSError:
        pass


def list_frames(folders: list[str]) -> list[Path]:
    """All the frames of the given folders (or files), sorted by name. In a folder holding raw or FITS frames, its JPEG,
    PNG and TIFF files stay aside (a camera's JPEGs next to its raw files are not more frames): those are the frames
    of a folder only when it holds no others (issue #57)."""
    out = []
    for f in folders:
        p = Path(f)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            found = [q for q in p.iterdir() if q.is_file() and frame_kind(q)]
            out += [q for q in found if frame_kind(q) != "colour"] or found
    return sorted(set(out))
