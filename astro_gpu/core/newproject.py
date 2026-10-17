"""Guesses for a new project from the folder of its light frames (no interface code: shared by the New project dialog
and the ComfyUI node): the darks / flats / bias / flat darks folders next to it, the project name, folders and target.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from .project import list_frames

GENERIC = {"raw", "raws", "raw2", "signal", "lights", "light", "brut", "bruts", "ok", "capture", "captures", "images",
           "serie1", "serie2", "series", "fits", "arw", "data"}
GUESSES = {"darks": ("dark", "darks", "Dark", "Darks", "DARK"), "flats": ("flat", "flats", "Flat", "Flats", "FLAT"),
           "bias": ("bias", "offset", "offsets", "Bias", "Offset"),
           "flat_darks": ("flatdark", "flat_dark", "flat_darks", "darkflat", "dark_flat", "FlatDarks")}
MONTHS = {"jan", "janv", "janvier", "january", "fev", "fevr", "fevrier", "february", "feb", "mar", "mars", "march",
          "avr", "avril", "april", "apr", "mai", "may", "juin", "june", "jun", "juil", "juillet", "july", "jul",
          "aout", "august", "aug", "sept", "sep", "septembre", "september", "oct", "octobre", "october", "nov",
          "novembre", "november", "dec", "decembre", "december"}
NOISE_WORDS = {"astro", "photo", "photos", "session", "nuit", "night", "test"}


def target_guess(folder_name: str) -> str:
    """Object name from a folder name: drop dates, numbers, month names and generic words (NGC7822 is kept)."""
    import unicodedata
    words = [w for w in re.split(r"[_\s-]+", folder_name) if w]
    keep = []
    for w in words:
        plain = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode().lower()
        if re.fullmatch(r"[0-9.]+", plain) or plain in MONTHS or plain in NOISE_WORDS:
            continue
        keep.append(w)
    return " ".join(keep)


def same_folder(a: str, b: str) -> bool:
    """Windows paths are case-insensitive: 'dark', 'Dark' and 'DARK' are the same folder."""
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def guess_calibration(lights_folder: str) -> dict[str, list[str]]:
    """The calibration folders next to (or inside) the lights folder, by their usual names, when they hold frames."""
    lp = Path(lights_folder)
    out: dict[str, list[str]] = {k: [] for k in GUESSES}
    for k, names in GUESSES.items():
        for nm in names:
            for base in (lp.parent, lp):
                c = base / nm
                if c.is_dir() and list_frames([str(c)]) and not any(same_folder(str(c), f) for f in out[k]):
                    out[k].append(str(c))
    return out


def guess_names(lights_folder: str) -> dict[str, str]:
    """Project name, project folder, scratch folder and target from the first meaningful ancestor of the lights
    folder (capture folder names such as 'lights' or 'raw' are skipped)."""
    named = Path(lights_folder)
    while named.name.lower() in GENERIC and named.parent != named:
        named = named.parent
    return {"name": named.name, "folder": str(named / "astro_gpu"),
            "scratch": str(Path.home() / "astro_gpu_scratch" / named.name), "target": target_guess(named.name)}


def space_text(lights: list[Path], scratch: str) -> str:
    """The calibrated frames cache needed on the scratch disk, against its free space."""
    if not lights or not scratch:
        return ""
    from ..io.frames import read_info
    inf = read_info(lights[0])
    need = len(lights) * inf.width * inf.height * 2 / 1e9
    root = Path(scratch).anchor or str(Path.home())
    free = shutil.disk_usage(root).free / 1e9
    return (f"Calibrated cache: about {need:.1f} GB on the scratch disk ({free:.0f} GB free)."
            + ("  Not enough space!" if need > free * 0.9 else ""))
