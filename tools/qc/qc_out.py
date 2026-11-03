"""Where the QC tools may write, and the proof that they write nowhere else.

Every tool of the v2 QC set (qc2.py and the qc_*.py modules) writes into one folder given as an argument (for example
F:/astro_projets/_qc/). That folder is refused when it is inside a project, v1 or v2, that the tools read, or inside
the folder of an image they read: the v1 tools wrote into ``<project>/results/`` (bulletin.py --sheet), and the old
deliverables must stay as they are.

``protect`` also arms the audit-hook write guard of core/import_v1 over the inputs for the length of a run, so that a
write there raises instead of happening (the module does not trust itself, nor the engine code it calls).

Layout under the output folder::

    sheets/     what judges may open: blind sheets and their brief (no label, no path, no old/new)
    _keys/      letter -> candidate, apart from the sheets; judges are never given this folder
    bulletin/   report cards      nonreg/  differences      pairs/  pairing tables      renders/  images made here
"""
from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from astro_gpu.core.import_v1 import WriteGuard

SHEETS, KEYS, BULLETIN, NONREG, PAIRS, RENDERS = "sheets", "_keys", "bulletin", "nonreg", "pairs", "renders"
PROJECT_FILES = ("project.astroproj", "project.json")


class OutputRefused(PermissionError):
    """The output folder is inside something the tools only read."""


def _norm(p: str | os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(p)))


def _inside(f: str, root: str) -> bool:
    return f == root or f.startswith(root.rstrip(os.sep) + os.sep)


def project_root_of(path: str | os.PathLike) -> Path | None:
    """The v1 or v2 project folder that holds ``path`` (the nearest parent with project.astroproj or project.json),
    or None."""
    p = Path(os.path.abspath(path))
    for d in [p, *p.parents]:
        if d.is_dir() and any((d / f).exists() for f in PROJECT_FILES):
            return d
    return None


def read_only_paths(inputs: Iterable[str | os.PathLike]) -> list[Path]:
    """What the run only reads: each input's project folder (every folder of it) and each file or folder given."""
    out: dict[str, Path] = {}
    for i in inputs:
        if not i:
            continue
        p = Path(os.path.abspath(i))
        root = project_root_of(p)
        for q in ([root] if root else []) + [p]:
            out.setdefault(_norm(q), q)
    return list(out.values())


def check_out(out: str | os.PathLike, inputs: Iterable[str | os.PathLike]) -> Path:
    """The absolute output folder, or ``OutputRefused`` when it is, or is inside, a project the run reads or the
    folder of an image it reads."""
    o = Path(os.path.abspath(out))
    no = _norm(o)
    for r in read_only_paths(inputs):
        nr = _norm(r)
        folder = nr if r.is_dir() else _norm(r.parent)      # an image: the folder it sits in counts too
        if _inside(no, nr) or _inside(no, folder):
            raise OutputRefused(f"the QC tools do not write under what they read: {o} is inside {r}")
    return o


@contextlib.contextmanager
def protect(out: str | os.PathLike, inputs: Iterable[str | os.PathLike]) -> Iterator[Path]:
    """``with protect(out, inputs) as out_dir:`` checks the output folder, creates it, and refuses every write under
    the inputs while the block runs."""
    inputs = [i for i in inputs if i]
    o = check_out(out, inputs)
    o.mkdir(parents=True, exist_ok=True)
    with WriteGuard(read_only_paths(inputs)):
        yield o


def sub(out: Path, name: str) -> Path:
    """A subfolder of the output folder (created)."""
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def slug(name: str) -> str:
    """A file-name-safe form of a name (the same rule as api.export_image's)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "image"
