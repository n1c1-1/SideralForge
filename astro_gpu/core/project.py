"""A processing project: one target, its input folders and everything computed from them.

Layout on disk::

    <project>/
        project.json        inputs, target, user parameter overrides, render settings
        work/<stage>/       outputs of each stage + manifest.json (+ _ckpt/ while a stage can be resumed)
        results/            exported images
        logs/events.jsonl   machine-readable journal (progress, metrics, previews) read by the GUI
        logs/journal.md     human-readable log

Large temporary data (the calibrated frames cache) lives in ``scratch_dir``, ideally on a fast local SSD.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

FORMAT = 1
RAW_EXT = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".orf", ".rw2", ".pef", ".srw", ".raf"}
FITS_EXT = {".fit", ".fits", ".fts"}


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


@dataclass
class ProjectConfig:
    name: str
    inputs: Inputs = field(default_factory=Inputs)
    target: Target = field(default_factory=Target)
    scratch_dir: str = ""
    reference_image: str = ""        # optional: a previous processing of the same data, for comparison
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    render: dict[str, Any] = field(default_factory=dict)
    devices: dict[str, str] = field(default_factory=dict)
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
        return cls(root)

    def _load(self) -> ProjectConfig:
        p = self.root / "project.json"
        if not p.exists():
            raise FileNotFoundError(f"not a project folder (no project.json): {self.root}")
        d = json.loads(p.read_text(encoding="utf-8"))
        return ProjectConfig(name=d["name"], inputs=Inputs(**d.get("inputs", {})), target=Target(**d.get("target", {})),
                             scratch_dir=d.get("scratch_dir", ""), reference_image=d.get("reference_image", ""),
                             params=d.get("params", {}), render=d.get("render", {}), devices=d.get("devices", {}),
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
        return self.work / stage

    # ------------------------------------------------------------------ parameters
    def overrides(self, stage: str) -> dict[str, Any]:
        return dict(self.config.params.get(stage, {}))

    def set_override(self, stage: str, name: str, value: Any) -> None:
        self.config.params.setdefault(stage, {})[name] = value
        self.save()

    def clear_override(self, stage: str, name: str) -> None:
        self.config.params.get(stage, {}).pop(name, None)
        self.save()


def list_frames(folders: list[str]) -> list[Path]:
    """All RAW / FITS files of the given folders (or files), sorted by name."""
    out = []
    for f in folders:
        p = Path(f)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [q for q in p.iterdir() if q.suffix.lower() in RAW_EXT | FITS_EXT]
    return sorted(set(out))
