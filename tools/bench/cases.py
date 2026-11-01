"""The quality bench: the five cases, their sources and the shared paths (tools/bench/*.py import this first).

Layout under BENCH (F:/astro_essais/bench):

    _home/                      USERPROFILE of our engine's runs: _home/.astro_gpu is its ASTRO_GPU_HOME (camera
                                profiles copied from the user's) and holds its catalogue cache (.astro_gpu/cache)
    _siril/config.ini           a copy of the user's Siril settings, given to siril-cli with -i (never written back)
    <case>/data/lights_express/ the frozen lights of each level (copies; Siril converts a whole folder)
    <case>/data/lights_base/
    <case>/data/{darks,flats,bias}/   every matching calibration frame (copies)
    <case>/manifest.json        files, sizes, camera, expected calibration
    <case>/siril_<level>/       Siril's work folder, its linear stacks (FITS) and finals (TIFF 16, PNG)
    <case>/ours_<level>/        our v2 project (workflow "Classic stack") and its exports
    <case>/measures_<level>.json, sheet_<level>.jpg

The sources are only read.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BENCH = Path("F:/astro_essais/bench")
HOME = BENCH / "_home"                         # USERPROFILE of our runs (Path.home() of the engine)
AG_HOME = HOME / ".astro_gpu"                  # ASTRO_GPU_HOME of our runs
USER_AG = Path(os.path.expanduser("~")) / ".astro_gpu"   # the user's: read only
REPO = Path(__file__).resolve().parents[2]
PYTHON = Path("C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe")
SIRIL = Path("C:/Program Files/Siril/bin/siril-cli.exe")
SIRIL_CFG_USER = Path(os.environ.get("LOCALAPPDATA", "")) / "siril" / "config.1.4.ini"
SIRIL_CFG = BENCH / "_siril" / "config.ini"

LEVELS = {"express": 30, "base": 100}
KINDS = ("darks", "flats", "bias")

D = "D:/astro_donnees_test"
#: name -> sources (folder, file suffixes) per kind, target (centre of the frames in degrees, approximate; obj: the
#: object's catalogue position, for the sheet's crop), optics
CASES: dict[str, dict] = {
    "heart": dict(
        lights="F:/photos/heart_nebula/raw/signal", darks="F:/photos/heart_nebula/raw/dark",
        flats="F:/photos/heart_nebula/raw/flat", ext=(".arw",),
        target="IC 1805", ra=40.76, dec=61.11, obj=(38.34, 61.45), focal=272.0, pixel=8.4,
        camera="Sony A7S (astro-modified), lossy ARW", scope="TS61EDPH refractor + 0.75x reducer (272 mm)",
        centre_from="the user's earlier Heart project (its astrometry), rounded"),
    "iris": dict(
        lights="D:/photos/2.3.25_Nebuleuse_iris/raw/signal", darks="D:/photos/2.3.25_Nebuleuse_iris/raw/dark",
        flats="D:/photos/2.3.25_Nebuleuse_iris/raw/flat", ext=(".arw",),
        target="NGC 7023", ra=315.92, dec=68.03, obj=(315.40, 68.163), focal=893.0, pixel=8.4,
        camera="Sony A7S (astro-modified), lossy ARW", scope="Newton 130/900 (~893 mm)",
        centre_from="the user's earlier Iris project (its astrometry), rounded"),
    "m51": dict(
        lights=f"{D}/canon6d_m51_n200/WhirlpoolGalaxy/lights", darks=f"{D}/canon6d_m51_n200/WhirlpoolGalaxy/darks",
        flats=f"{D}/canon6d_m51_n200/WhirlpoolGalaxy/flats", bias=f"{D}/canon6d_m51_n200/WhirlpoolGalaxy/biases",
        ext=(".cr2",), target="M 51", ra=202.47, dec=47.195, obj=(202.4696, 47.1952), focal=1000.0, pixel=6.54,
        camera="Canon EOS 6D (CR2)", scope="Newton Sky-Watcher 200/1000",
        centre_from="M 51's catalogue position"),
    "m8": dict(
        lights=f"{D}/siril_asi2600mc_m8/Brutes_180s", darks=f"{D}/siril_asi2600mc_m8/Darks_180s",
        flats=f"{D}/siril_asi2600mc_m8/Flats_3s", bias=f"{D}/siril_asi2600mc_m8/Offsets_3s", ext=(".fit", ".fits"),
        target="M 8", ra=271.295, dec=-23.497, obj=(270.904, -24.387), focal=388.0, pixel=3.76,
        camera="ZWO ASI2600MC Pro (FITS, cooled)", scope="Takahashi Temma 2 (388 mm)",
        centre_from="the frames' FITS header (RA, DEC)",
        note="only the 15 broadband lights: the 20 dual-band (HOO) lights have another filter and other flats; "
             "a Classic stack does not mix them"),
    "seestar": dict(
        lights=f"{D}/seestar_s50_s30_m31_m42/VergleicheS50vsS30/S50/M 31/M 31_sub1", ext=(".fit", ".fits"),
        target="M 31", ra=11.0, dec=41.327, obj=(10.6847, 41.2690), focal=250.0, pixel=2.9,
        camera="ZWO Seestar S50 (FITS, IRCUT filter)", scope="Seestar S50 (250 mm, alt-azimuth)",
        centre_from="the frames' FITS header (RA, DEC)",
        note="no calibration frames exist for this set (the Seestar takes none)"),
}


def data_dir(case: str) -> Path:
    return BENCH / case / "data"


def lights_dir(case: str, level: str) -> Path:
    return data_dir(case) / f"lights_{level}"


def kind_dir(case: str, kind: str) -> Path:
    return data_dir(case) / kind


def levels_of(case: str) -> list[str]:
    """The levels a case has: a level with the same lights as a smaller one is not repeated (m8: 15 lights)."""
    import json
    m = json.loads((BENCH / case / "manifest.json").read_text(encoding="utf-8"))
    return [lv for lv in LEVELS if not m["levels"][lv].get("same_as")]


def utf8_console() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
