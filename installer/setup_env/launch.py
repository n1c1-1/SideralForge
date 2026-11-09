"""What the Start menu and desktop shortcuts run (with the private Python's pythonw): finishes the installation if a
former one was stopped, then starts FLOAT, and says so if it cannot start.

    pythonw launch.py [project folder]
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import setup_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DROP = ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX")


def alert(text: str):
    try:
        ctypes.windll.user32.MessageBoxW(0, text, "FLOAT", 0x10)
    except (AttributeError, OSError):
        pass


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not setup_env.is_ready(ROOT):
        rc = setup_env.main(["--root", str(ROOT)])
        if rc != 0:
            return rc
    # the log is a file: without this, a name outside the Windows code page (a project, a user) would stop a print
    env = {**{k: v for k, v in os.environ.items() if k not in DROP}, "PYTHONIOENCODING": "utf-8"}
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / "float.log"
    with open(log_path, "wb") as log:
        # python.exe in a hidden console (not pythonw): the computing workers it starts then have a console of their
        # own and their child processes never open a window of their own
        rc = subprocess.call([str(ROOT / "env" / "Scripts" / "python.exe"), "-m", "astro_gpu.client", *args], env=env,
                             stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if rc != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        fr = setup_env.windows_language() == "fr"
        alert(("FLOAT s'est arrêté. Détails : " if fr else "FLOAT stopped. Details: ") + f"{log_path}\n\n{tail}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
