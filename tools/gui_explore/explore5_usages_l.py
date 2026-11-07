"""usages L (English): another project opened from Home's card while the current one has edits: the prompt, Cancel
keeps the current project and its edits, Don't save opens the other one; File > Close the same."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore5_usages_common import EXC, U, _, log, start, wait  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

start(600)
rep = sys.argv[1] if len(sys.argv) > 1 else "1"
u = U("en")
ex = u.ex
REAL = {"on": False}
QApplication.platformName = staticmethod(lambda: "windows" if REAL["on"] else "offscreen")
a, _c = u.new_from_home(f"Other_{rep}", "classic")
u.save()
ex.w.close_project()
b, _c = u.new_from_home(f"Current_{rep}", "classic")
u.save()
u.place("denoise")
t0 = ex.w.windowTitle()
REAL["on"] = True
u.box_answer = [_("Cancel")]
n = len(u.boxes)
ex.w.open_project(a)
wait(ex, 0.8)
log("L cancel", u.boxes[n:], "project", ex.w.project.name if ex.w.project else None, "title", t0, "->",
    ex.w.windowTitle(), "screen home?", ex.w.screens.currentWidget() is ex.w.home if hasattr(ex.w, "home") else None)
ex.shot(f"L{rep}_after_cancel")
u.box_answer = [_("Don't save")]
n = len(u.boxes)
ex.w.open_project(a)
wait(ex, 0.8)
log("L don't save", u.boxes[n:], "project", ex.w.project.name if ex.w.project else None)
u.place("denoise")
u.box_answer = [_("Cancel")]
n = len(u.boxes)
r = ex.w.close_project()
wait(ex, 0.5)
log("L File>Close cancel", r, u.boxes[n:], "project", ex.w.project.name if ex.w.project else None,
    "screen", ex.w.screens.currentWidget().__class__.__name__)
REAL["on"] = False
log("EXC", len(EXC), "\n".join(e[-1200:] for e in EXC))
ex.w.s._dirty = False
ex.w.close()
