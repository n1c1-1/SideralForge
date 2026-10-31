"""The user's settings, kept between runs (alt3/user_settings.json): language, folders, compute, catalogs, file
formats, display, master library, spec mode."""
from __future__ import annotations

import json
import os
from pathlib import Path

# ALT3_SETTINGS: another file, so that automatic tests never touch the user's settings
FILE = Path(os.environ.get("ALT3_SETTINGS") or Path(__file__).resolve().parents[1] / "user_settings.json")

DEFAULTS = {
    "language": "en",
    "spec_mode": False,
    # folders and disk
    "workspace": "F:/astro_projects",
    "cache_dir": "C:/astro_cache",
    "temp_dir": "C:/Users/Nicolas/AppData/Local/Temp/astro",
    "cache_limit_gb": 200,
    # compute
    "device": "auto",                 # auto | gpu0 | cpu
    "threads": 0,                     # 0 = all cores but two
    "memory_limit_gb": 24,
    "benchmark": None,                # result of the machine benchmark
    # catalogs and astrometry
    "gaia": "local",                  # local | online
    "gaia_path": "F:/catalogs/gaia_dr3",
    "solver": "internal",             # internal | astap | astrometry.net
    "survey_cache": "F:/catalogs/surveys",
    # file formats
    "format": "FITS",                 # FITS | XISF | TIFF
    "bits": "32-bit float",
    "compression": "None",
    # display
    "auto_stretch": True,
    "units": "arcsec",                # arcsec | px
    # master library
    "masters_dir": "F:/astro_masters",
    "master_tolerance_temp": 2.0,     # °C
    "master_tolerance_days": 180,
}

_data: dict = {}


def load():
    global _data
    _data = dict(DEFAULTS)
    if FILE.exists():
        try:
            _data.update(json.loads(FILE.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return _data


def get(key):
    if not _data:
        load()
    return _data.get(key, DEFAULTS.get(key))


def set(key, value):                  # noqa: A001 - a settings store
    if not _data:
        load()
    _data[key] = value
    try:
        FILE.write_text(json.dumps(_data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def spec() -> bool:
    """Spec mode: shows what the engine still has to build (◇ marks, estimated durations, mockup notes)."""
    return bool(get("spec_mode"))
