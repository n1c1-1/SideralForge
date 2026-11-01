"""P8, the user's projects: the shared settings of the tools/projects scripts (DESIGN §6.3 P8).

Everything P8 writes goes under F:/astro_projets/:

    _home/                 the pilot ASTRO_GPU_HOME (a copy of the camera profiles, the seeded timing history)
    _raw/<project>/<kind>/ the raw frames that were not on F:, copied (frame files only, dates kept)
    _p8/                   reports, no-write proofs, queue definitions, states and logs
    <project>/             the v2 projects

The user's v1 projects, their raws and their learned star layers are only read. Heart and NGC 7822 keep their raws
in place on F:; the others' raws are copied to _raw/ and the import's path map points the projects at the copies.

Import this module before anything of astro_gpu: ``use_home()`` sets ASTRO_GPU_HOME, which astro_gpu reads at import.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_HERE = os.path.normcase(str(Path(__file__).resolve().parent))
# this folder holds queue.py, which would hide the standard library's queue module (concurrent.futures imports it):
# once p8 is imported, the folder leaves sys.path
sys.path[:] = [x for x in sys.path if os.path.normcase(os.path.abspath(x or ".")) != _HERE]
for _s in (sys.stdout, sys.stderr):           # a Windows console is cp1252: '≈' and the French reasons print anyway
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
WS = Path("F:/astro_projets")
HOME = WS / "_home"                      # the pilot ASTRO_GPU_HOME
RAW = WS / "_raw"
P8 = WS / "_p8"
QUEUES = P8 / "queues"
PYTHON = Path("C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe")
USER_HOME = Path.home() / ".astro_gpu"   # the user's ASTRO_GPU_HOME (no ASTRO_GPU_HOME variable is set)

G28 = "G:/Astro/2024-12-28_divers_astro"
OLD28 = "F:/backup_pc/photos/28.12.24"   # the dead place the three 28.12.24 projects name
IRIS_RAW = "D:/photos/2.3.25_Nebuleuse_iris/raw"


@dataclass
class Copy:
    """Raw frames to copy: the files of ``sources`` (folders; their frame files) or, with ``listed``, only the files
    the project lists under ``old`` (taken from the first source folder by name). ``old`` is the place the v1
    project names (the key of the path map); the copy goes to _raw/<project>/<kind>/."""

    old: str
    kind: str
    sources: list[str]
    listed: bool = False


@dataclass
class Proj:
    name: str                            # the v2 folder name under WS
    old: str                             # the v1 project folder (read only)
    copies: list[Copy] = field(default_factory=list)
    extra_map: dict[str, str] = field(default_factory=dict)    # moved inputs that are not copied (read in place)

    @property
    def new(self) -> Path:
        return WS / self.name

    def raw_dir(self, kind: str) -> Path:
        return RAW / self.name / kind

    def path_map(self) -> dict[str, str]:
        out = {c.old: self.raw_dir(c.kind).as_posix() for c in self.copies}
        out.update(self.extra_map)
        out.update({old: staged.as_posix() for old, _, staged in staged_inputs(self)})
        return out


def _old_abs(root: str, path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else Path(root) / p)


def staged_inputs(proj: Proj) -> list[tuple[str, Path, Path]]:
    """(the path the v1 project names, the file read, its staged copy) of the reference image and the earlier star
    layers whose date a copy onto F: cannot keep.

    F: is exFAT: a copy's modification time is rounded up to 2 s there, so a file copied from NTFS changes its
    whole-second date, hence its file_id (name, size, date) and the keys that hold it, and import_v1 refuses with
    "the keys changed when the inputs moved to inputs/". Such files are first copied to _raw/<project>/inputs/ (their
    date is then on exFAT's grid, and the import's own copy into inputs/ keeps it); the path map points at them. The
    content is the same file: no result changes."""
    import json
    d = json.loads((Path(proj.old) / "project.json").read_text(encoding="utf-8"))
    names = [d.get("reference_image") or ""]
    for params in [d.get("params") or {}] + [(v or {}).get("params") or {} for v in (d.get("variants") or {}).values()]:
        v = str((params.get("starless") or {}).get("learned_stars") or "").strip()
        if v and v.lower() not in ("none", "off"):
            names.append(re.sub(r" \(run [0-9a-fA-F]*\)$", "", v))
    out = []
    for n in dict.fromkeys(x for x in names if x):
        old = _old_abs(proj.old, n)
        src = Path(old)
        for k, v in proj.extra_map.items():                  # a moved reference (M31's, now on G:)
            ko = os.path.normcase(os.path.normpath(k))
            if os.path.normcase(os.path.normpath(old)).startswith(ko + os.sep):
                src = Path(v) / os.path.normpath(old)[len(ko):].lstrip("\\/")
        if src.is_file() and src.stat().st_mtime_ns % 2_000_000_000:
            out.append((old, src, RAW / proj.name / "inputs" / src.name))
    return out


def _darks28() -> Copy:
    return Copy(f"{OLD28}/957MSDCF", "darks", [f"{G28}/957MSDCF"], listed=True)


def _flats28() -> Copy:
    return Copy(f"{IRIS_RAW}/flat", "flats", [f"{IRIS_RAW}/flat"])


PROJECTS: list[Proj] = [
    Proj("Heart", "F:/photos/heart_nebula/astro_gpu"),                       # raws on F:, read in place
    Proj("NGC7822", "F:/photos/NGC7822/astro_gpu"),                          # raws on F:, read in place
    Proj("Iris", "C:/Users/Nicolas/astro_gpu_scratch/projects/Iris",
         [Copy(f"{IRIS_RAW}/signal", "lights", [f"{IRIS_RAW}/signal"]),
          Copy(f"{IRIS_RAW}/dark", "darks", [f"{IRIS_RAW}/dark"]),
          Copy(f"{IRIS_RAW}/flat", "flats", [f"{IRIS_RAW}/flat"])]),
    Proj("M33", "C:/Users/Nicolas/astro_gpu_scratch/projects/M33",
         [Copy("D:/astro_photo/22_Aout_25_Triangle/Raw2", "lights", ["D:/astro_photo/22_Aout_25_Triangle/Raw2"]),
          Copy("D:/astro_photo/22_Aout_25_Triangle/Flat", "flats", ["D:/astro_photo/22_Aout_25_Triangle/Flat"])]),
    Proj("Rosette", "E:/astro_gpu/projects/Rosette",
         [Copy(f"{OLD28}/Rosette/signal", "lights", [f"{G28}/Rosette/signal"]), _darks28(), _flats28()]),
    Proj("Horsehead", "E:/astro_gpu/projects/Horsehead",
         [Copy(f"{OLD28}/HorseHead/signal", "lights", [f"{G28}/HorseHead/signal"]), _darks28(), _flats28()]),
    # one folder in its project.json, two on G: now: merged back into one, so the old order is kept
    Proj("FlamingStar", "E:/astro_gpu/projects/FlamingStar",
         [Copy(f"{OLD28}/Etoile_flamboyante/signal", "lights",
               ["G:/Astro/2024-12-27_Etoile-Flamboyante/raw", f"{G28}/Etoile_flamboyante/signal"]),
          _darks28(), _flats28()]),
    Proj("M31", "E:/astro_gpu/projects/M31",
         [Copy("F:/backup_pc/photos/24.08.23/signal", "lights", ["G:/Astro/2023-08-24_M31_top/signal"]),
          Copy("F:/backup_pc/photos/24.08.23/flat", "flats", ["G:/Astro/2023-08-24_M31_top/flat"])],
         # the reference image: the import copies it into the project's inputs/
         {"F:/backup_pc/photos/24.08.23/traitement": "G:/Astro/2023-08-24_M31_top/traitement"}),
    Proj("Wizard", "E:/astro_gpu/projects/Wizard",
         [Copy("E:/Astro-photo/Sorciere/raw/signal", "lights", ["E:/Astro-photo/Sorciere/raw/signal"]),
          Copy("E:/Astro-photo/Sorciere/raw/dark", "darks", ["E:/Astro-photo/Sorciere/raw/dark"]),
          Copy("E:/Astro-photo/Sorciere/raw/flat", "flats", ["E:/Astro-photo/Sorciere/raw/flat"])]),
]
BY_NAME = {p.name: p for p in PROJECTS}

#: P6's dry-run (JOURNAL, tests/test_import_v1.py REAL): the stages adopted in the first workflow
PREFIX = ["inventory", "calibrate", "register", "stack", "astrometry", "reference", "catalogs", "surveys", "color",
          "denoise", "optics"]
NO_REF = [s for s in PREFIX if s != "reference"]
EXPECTED_ADOPTED = {"Heart": PREFIX, "NGC7822": ["inventory"], "Iris": PREFIX, "M33": ["inventory"],
                    "Rosette": PREFIX, "Horsehead": PREFIX, "FlamingStar": PREFIX, "M31": NO_REF, "Wizard": NO_REF}


def use_home() -> None:
    """Point ASTRO_GPU_HOME at the pilot home, before astro_gpu is imported; put the repository on sys.path."""
    if "astro_gpu" in sys.modules and os.environ.get("ASTRO_GPU_HOME") != str(HOME):
        raise RuntimeError("astro_gpu is already imported with another ASTRO_GPU_HOME")
    os.environ["ASTRO_GPU_HOME"] = str(HOME)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def protected(proj: Proj) -> list[str]:
    """Everything of the user's that a P8 step only reads for this project: the v1 folder, its scratch, inputs,
    reference and star layers (import_v1.folders_of), and the raw folders the copies come from."""
    from astro_gpu.core import import_v1 as iv
    out = iv.folders_of(proj.old)
    for c in proj.copies:
        out += c.sources
    out += list(proj.extra_map.values())
    return list(dict.fromkeys(out))


def under_ws(path: str | Path) -> bool:
    """Whether a path is inside F:/astro_projets (the only place P8 writes)."""
    a = os.path.normcase(os.path.abspath(path))
    w = os.path.normcase(os.path.abspath(WS))
    return a == w or a.startswith(w + os.sep)
