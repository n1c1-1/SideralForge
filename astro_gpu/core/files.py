"""Atomic file writes that survive Windows readers: ``replace_file`` (``os.replace`` retried while a reader holds the
target) and ``write_json`` (a JSON file written beside its target, then put in its place).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def replace_file(src: Path, dst: Path, attempts: int = 10) -> None:
    """os.replace of a file over another, retried for about 5 s on PermissionError: on Windows a file cannot be
    replaced while a reader has it open (the client reading a manifest, an antivirus, the indexer), usually for a
    moment only."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(min(0.05 * 2 ** i, 1.0))


def write_json(path: str | Path, obj: Any, indent: int | None = 1, default: Any = str) -> None:
    """Write a JSON file atomically: through a temporary file of its own (process id and a random tag, so two writers
    never share one) and ``replace_file``. A reader never sees half a file, and no temporary file is left behind,
    also when the write fails."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=indent, default=default), encoding="utf-8")
        replace_file(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
