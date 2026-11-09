"""A project not named yet, and the recovery after a crash.

"New project" opens the graph at once: the project lives in a folder of its own under the application's settings folder,
as "Untitled", until its name and folder are asked (the first Save or the first Run: ``Main.name_project``). Closed
without edits it leaves nothing behind.

Recovery, for every project: some seconds after an edit, a copy of the definition is written here, apart from the
project's own file (which only Save writes). It goes when the project is saved or closed normally; found at start, a crash
left it: Home says so, and offers to reopen it.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from ..core.files import write_json
from . import prefs


def _base() -> Path:
    return prefs.FILE.parent if prefs.FILE is not None else Path(tempfile.gettempdir()) / "astro_gpu"


def untitled_folder() -> Path:
    return _base() / "untitled"


def recovery_folder() -> Path:
    return _base() / "recovery"


def new_root() -> Path:
    """The folder of a new project without a name."""
    return untitled_folder() / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def is_untitled(root) -> bool:
    return Path(root).parent == untitled_folder()


def discard(root) -> None:
    """A project without a name goes: its files, then its folder (nothing else is ever deleted here, and only directly
    in that folder: what cannot go stays)."""
    root = Path(root)
    if not is_untitled(root):
        return
    try:
        for f in root.iterdir():
            if f.is_file():
                f.unlink()
        root.rmdir()
    except OSError:
        pass


def _file_of(root) -> Path:
    return recovery_folder() / (hashlib.sha1(Path(root).as_posix().lower().encode("utf-8")).hexdigest() + ".json")


def write_recovery(root, name: str, definition: dict) -> None:
    """The copy of a project with edits not saved (a disk that refuses it costs only the copy)."""
    try:
        recovery_folder().mkdir(parents=True, exist_ok=True)
        write_json(_file_of(root), {"root": Path(root).as_posix(), "name": name,
                                    "saved": f"{datetime.now():%Y-%m-%d %H:%M}", "definition": definition},
                   indent=None, default=None)
    except OSError:
        pass


def clear_recovery(root) -> None:
    try:
        _file_of(root).unlink()
    except OSError:
        pass


def recoveries() -> list[dict]:
    """What a crash left, the latest first; a file that cannot be read is left out."""
    out = []
    folder = recovery_folder()
    for f in folder.glob("*.json") if folder.is_dir() else ():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("root") and isinstance(d.get("definition"), dict):
            out.append(d)
    return sorted(out, key=lambda d: str(d.get("saved")), reverse=True)


def clean_orphans(keep=None) -> None:
    """The folders of untitled projects that no recovery names (left by a crash before any edit), but ``keep``."""
    folder = untitled_folder()
    named = {Path(d["root"]) for d in recoveries()}
    for root in list(folder.iterdir()) if folder.is_dir() else ():
        if root.is_dir() and root not in named and root != keep:
            discard(root)
