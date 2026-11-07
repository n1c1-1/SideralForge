"""usages H: where Home's New project puts the project after frames were added (its dialog shot); a project folder
renamed outside, seen from Home (its card, a click on it)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore5_usages_common import EXC, Dispatch, U, log, start, wait  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

start(600)
rep = sys.argv[1] if len(sys.argv) > 1 else "1"
u = U("fr")
ex = u.ex
from astro_gpu.client import prefs  # noqa: E402

# 1. frames added (lights, then darks) in a first project, then Home > New project: where?
lights = ex.raws(f"Ou_{rep}/lights", 4)
darks = ex.raws(f"Ou_{rep}/darks", 3, start=8000)
root, _c = u.new_from_home(f"Premier_{rep}", "blank")
u.place("session")
u.add_frames(1, lights, 0)
u.add_frames(1, darks, 1)
u.save()
ex.w.close_project()
wait(ex, 0.5)
cap = {}


def npd(d):
    d.name.setText(f"Second_{rep}")
    ex.settle(3)
    cap["folder"] = d.folder().as_posix()
    cap["label"] = d.folder_lb.text() if hasattr(d.folder_lb, "text") else ""
    cap["tip"] = d.folder_lb.toolTip()
    ex.shot(f"H{rep}_new_project_dialog", d)
    d.reject()


disp = Dispatch(ex, NewProjectDialog=npd)
ex.w.new_project()
wait(ex, 1.0)
disp.stop()
log("H new project proposed in", cap, "last_folder", prefs.last_folder())

# 2. a project in the projects' folder, renamed outside, seen from Home
prefs.set("last_folder", ex.ws.as_posix())
root, _c = u.new_from_home(f"Deplace_{rep}", "classic")
u.save()
ex.w.close_project()
wait(ex, 0.5)
os.replace(root, root.with_name(root.name + "_ailleurs"))
ex.w.show_home()
wait(ex, 4.0)
from astro_gpu.client.home import ProjectCard  # noqa: E402
cards = [c for c in ex.w.findChildren(ProjectCard) if c.isVisible()]
log("H cards", [(c.info.name, getattr(c.info, "root", None)) for c in cards])
ex.shot(f"H{rep}_home_after_move")
mine = [c for c in cards if c.info.name == root.name]
n = len(ex.popups)
if mine:
    ex.click(mine[0])
    wait(ex, 1.0)
    log("H click", ex.popups[n:], ex.w.project)
    ex.shot(f"H{rep}_after_click")
    cards = [c for c in ex.w.findChildren(ProjectCard) if c.isVisible()]
    log("H cards after", [c.info.name for c in cards])
log("EXC", len(EXC), "\n".join(e[-1200:] for e in EXC))
ex.w.close()
