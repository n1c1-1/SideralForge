"""The user's settings, kept between runs: language, folders, compute, catalogs, display, spec mode, recent projects.

They live in %APPDATA%/astro_gpu/settings.json. ASTRO_GPU_SETTINGS names another file, so that tests and scripted
walkthroughs never touch the user's settings; under pytest, without it, nothing is read or written.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ENV = "ASTRO_GPU_SETTINGS"
RECENT_MAX = 12

DEFAULTS = {
    "language": "en",
    "spec_mode": False,
    "recent": [],                     # [{"path", "name", "target"}], most recent first
    # folders and disk. No workspace (issue #31): a project lives where the user puts it
    # the folder the file dialogs open in: the last one used (issues #31, #32)
    "last_folder": "",
    # the user's own templates (issue #47): {key, tab, name, text, source, blocks}, made with Save as template…
    "templates": [],
    # the sections of Home (Settings › Folders), under the recent projects: each lists the projects of a folder and of
    # its sub-folders. "label" and "note" are English (translated where shown) or typed by the user; "open": shown open.
    # None at first (the night test pass: a folder of the development machine was shown to every new user)
    "home_sections": [],
    "cache_dir": (Path.home() / "astro_gpu_scratch").as_posix(),   # calibrated frames: <cache_dir>/<project>
    "temp_dir": (Path(tempfile.gettempdir()) / "astro_gpu").as_posix(),
    "cache_limit_gb": 200,
    # compute
    "device": "auto",                 # auto | cuda:<n> | xpu:<n> | mps | privateml:<n> | cpu
    # what the card test decided: {device (or "auto"): profile}, core/selftest.py; {}: not tested yet
    "compute_profiles": {},
    "threads": 0,                     # 0 = all cores but two
    "memory_limit_gb": 24,
    # catalogs and astrometry
    "solver": "internal",             # internal | astap | astrometry.net
    "survey_cache": "",
    # file formats
    "format": "FITS",                 # FITS | XISF
    "bits": "32-bit float",
    "compression": "None",
    "keep_frames": True,
    # display
    "auto_stretch": True,
    "units": "arcsec",                # arcsec | px
    # the window on a small screen (issue #44): auto | compact | comfortable; the user's panels per size range
    "layout_density": "auto",
    "layout": {},
    # master library
    "masters_dir": "",
    "master_tolerance_temp": 2.0,     # °C
    "master_tolerance_days": 180,
}

_data: dict = {}
FILE: Path | None = None


def settings_file() -> Path | None:
    """The settings file: ASTRO_GPU_SETTINGS, else %APPDATA%/astro_gpu/settings.json; None under pytest without
    ASTRO_GPU_SETTINGS (the tests never read or write the user's file)."""
    env = os.environ.get(ENV)
    if env:
        return Path(env)
    if "PYTEST_CURRENT_TEST" in os.environ:
        return None
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def load():
    global _data, FILE
    FILE = settings_file()
    _data = dict(DEFAULTS)
    _data["recent"] = []
    if FILE is not None and FILE.exists():
        try:
            _data.update(json.loads(FILE.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return _data


def system_language() -> str:
    """The language of the system's interface when the program has it ("fr"), else "en": the language of a first
    launch (the night test pass: a French user started in English and had to find Edit > Settings > Language)."""
    import locale
    try:
        import ctypes
        name = locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage(), "")
    except (AttributeError, OSError):                 # not Windows
        name = locale.getlocale()[0] or ""
    return "fr" if name.lower().startswith("fr") else "en"


def get(key):
    if not _data:
        load()
    return _data.get(key, DEFAULTS.get(key))


def set(key, value):                  # noqa: A001 - a settings store
    if not _data:
        load()
    _data[key] = value
    save()


def save():
    """Write the settings atomically (nothing when there is no settings file: pytest without ASTRO_GPU_SETTINGS)."""
    if FILE is None:
        return
    try:
        FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = FILE.with_name(FILE.name + ".tmp")
        tmp.write_text(json.dumps(_data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, FILE)
    except OSError:
        pass


def spec() -> bool:
    """Spec mode: shows what the engine still has to build (◇ marks), disabled."""
    return bool(get("spec_mode"))


def last_folder() -> str:
    """Where a file dialog opens: the last folder used in one (issues #31, #32: every dialog started from a fixed
    workspace, so the flats were looked for again from there after the lights), else the user's home folder."""
    cand = get("last_folder")
    return str(cand) if cand and Path(cand).is_dir() else str(Path.home())


def remember_folder(path) -> None:
    """A file or folder chosen in a dialog: its folder is where the next dialog opens."""
    if not path:
        return
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    if str(folder) and folder.is_dir():
        set("last_folder", folder.as_posix())


def recent() -> list[dict]:
    """The recently opened projects, most recent first: [{"path", "name", "target"}]."""
    return [dict(r) for r in get("recent") or [] if isinstance(r, dict) and r.get("path")]


def add_recent(path: str, name: str, target: str = ""):
    path = Path(path).as_posix()
    rows = [r for r in recent() if Path(r["path"]).as_posix().lower() != path.lower()]
    set("recent", [{"path": path, "name": name, "target": target}] + rows[:RECENT_MAX - 1])


def drop_recent(path: str):
    path = Path(path).as_posix().lower()
    set("recent", [r for r in recent() if Path(r["path"]).as_posix().lower() != path])
