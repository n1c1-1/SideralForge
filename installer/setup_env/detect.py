"""Which graphics card this computer has, and which PyTorch build to install for it.

Standard library only: this runs from the installer's private Python, before anything else is installed.

The rule (everybody gets a build that was tested, so one build per family):
* an NVIDIA card with a driver of at least ``CUDA_MIN_DRIVER`` gets the CUDA build of the lock file: its wheel carries
  the CUDA runtime, only the driver is needed, the CUDA toolkit is never installed. A newer driver also gets this
  build (a driver runs older CUDA versions): 2.11.0+cu128 is the build the app's tests pass on.
* no NVIDIA card, a driver too old or unreadable, Intel, AMD: the CPU build, with a plain message.

``detect()`` never raises: whatever goes wrong here ends in the CPU build, the installation goes on.
``FLOAT_TEST_GPU`` replaces the detection (``none``, ``nvidia:610.88``, ``amd:Radeon RX 7600``, ``intel:Arc``):
end-to-end tests of the installer choose the path they test with it.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass

#: NVIDIA's minimum Windows driver for CUDA 12 (table 2 of the CUDA toolkit release notes)
CUDA_MIN_DRIVER = (528, 33)
#: the lock file's build for a recent enough NVIDIA card, and for everything else
CUDA_BUILD = "cu128"
CPU_BUILD = "cpu"
TEST_VARIABLE = "FLOAT_TEST_GPU"
#: the longest path Windows opens without long paths, and the longest path of an installed file below the install
#: folder (173 measured on a real installation of the CUDA build, the longest; a little margin)
MAX_PATH = 259
MAX_RELATIVE_PATH = 180

DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


@dataclass(frozen=True)
class Gpu:
    kind: str                                # nvidia | amd | intel | other
    name: str
    driver: tuple[int, int] | None = None    # NVIDIA's numbering (610, 88); None when unknown


@dataclass(frozen=True)
class Choice:
    build: str                               # a key of the lock file's "builds"
    why: str                                 # ok | no-card | old-driver | no-driver | other-vendor | error
    name: str = ""                           # the card the message speaks of
    driver: str = ""                         # its driver, as "610.88"


def root_length(root) -> int:
    return len(str(root))


def root_limit() -> int:
    """The longest install folder that keeps every installed file under Windows' 260 characters, whatever the long
    paths setting: neither the installer nor the uninstaller can handle longer ones (the uninstaller left files behind
    in a test). The installer's folder page uses the same figure (build.py passes it to Inno)."""
    return MAX_PATH - MAX_RELATIVE_PATH


def kind_of(name: str) -> str:
    low = name.lower()
    if any(w in low for w in ("nvidia", "geforce", "quadro", "tesla")):
        return "nvidia"
    if "amd" in low or "radeon" in low:
        return "amd"
    if "intel" in low or "arc" in low.split():
        return "intel"
    return "other"


def parse_driver(text: str) -> tuple[int, int] | None:
    """(610, 88) from "610.88", the way nvidia-smi writes it."""
    m = re.fullmatch(r"\s*(\d{2,4})\.(\d{1,3})\s*", text or "")
    return (int(m[1]), int(m[2])) if m else None


def parse_registry_driver(text: str) -> tuple[int, int] | None:
    """NVIDIA's numbering from Windows' driver version: the last five digits, 32.0.15.8088 -> (580, 88)."""
    digits = "".join(re.findall(r"\d+", ".".join((text or "").split(".")[-2:])))
    return (int(digits[-5:-2]), int(digits[-2:])) if len(digits) >= 5 else None


def driver_text(driver: tuple[int, int] | None) -> str:
    return f"{driver[0]}.{driver[1]:02d}" if driver else ""


def _nvidia_smi_candidates() -> list[str]:
    out = []
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    for p in (os.path.join(root, "System32", "nvidia-smi.exe"),
              os.path.join(os.environ.get("ProgramW6432") or r"C:\Program Files", "NVIDIA Corporation", "NVSMI",
                           "nvidia-smi.exe")):
        if os.path.isfile(p):
            out.append(p)
    return out or ["nvidia-smi"]


def _smi_gpus() -> list[Gpu]:
    for exe in _nvidia_smi_candidates():
        try:
            r = subprocess.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True,
                               text=True, errors="replace", timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode != 0:
            continue
        gpus = []
        for line in r.stdout.splitlines():
            name, _, driver = line.rpartition(",")
            if name.strip():
                gpus.append(Gpu("nvidia", name.strip(), parse_driver(driver)))
        if gpus:
            return gpus
    return []


def _registry_gpus() -> list[Gpu]:
    try:
        import winreg
    except ImportError:
        return []
    gpus = []
    try:
        cls = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, DISPLAY_CLASS)
    except OSError:
        return []
    with cls:
        for i in range(32):
            try:
                sub = winreg.OpenKey(cls, f"{i:04d}")
            except OSError:
                continue
            with sub:
                try:
                    name = str(winreg.QueryValueEx(sub, "DriverDesc")[0])
                    version = str(winreg.QueryValueEx(sub, "DriverVersion")[0])
                except OSError:
                    continue
            kind = kind_of(name)
            if name and "microsoft basic" not in name.lower():
                gpus.append(Gpu(kind, name, parse_registry_driver(version) if kind == "nvidia" else None))
    return gpus


def _test_gpus(value: str) -> list[Gpu]:
    if value == "none":
        return []
    kind, _, rest = value.partition(":")
    return [Gpu("nvidia", "Test NVIDIA", parse_driver(rest))] if kind == "nvidia" else [Gpu(kind, rest or f"Test {kind}")]


def choose(gpus: list[Gpu]) -> Choice:
    """The build for these cards: CUDA when an NVIDIA card has a recent enough driver."""
    nvidia = [g for g in gpus if g.kind == "nvidia"]
    if nvidia:
        best = max(nvidia, key=lambda g: g.driver or (0, 0))
        if best.driver is None:
            return Choice(CPU_BUILD, "no-driver", best.name)
        if best.driver >= CUDA_MIN_DRIVER:
            return Choice(CUDA_BUILD, "ok", best.name, driver_text(best.driver))
        return Choice(CPU_BUILD, "old-driver", best.name, driver_text(best.driver))
    others = [g for g in gpus if g.kind in ("amd", "intel")]
    if others:
        return Choice(CPU_BUILD, "other-vendor", others[0].name)
    return Choice(CPU_BUILD, "no-card")


def detect(env=None) -> Choice:
    """The choice for this computer, never an exception: a failure of the detection is the CPU build."""
    env = os.environ if env is None else env
    try:
        test = env.get(TEST_VARIABLE)
        if test:
            return choose(_test_gpus(test))
        gpus = _smi_gpus()
        return choose(gpus or _registry_gpus())
    except Exception:  # noqa: BLE001 - installation never fails because of the card
        return Choice(CPU_BUILD, "error")


if __name__ == "__main__":
    print(detect())
    sys.exit(0)
