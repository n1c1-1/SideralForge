"""Probe: the Combine block Post gives by default (block_actions area)."""
import faulthandler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(250, exit=True)
from explore_blocks_lib import EXC, TOASTS, drawn, panel_info, state  # noqa: E402
from harness import Explorer  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

ex = Explorer("_probe_blocks")
CHAIN = ["calibration", "registration", "selection", "stacking", "astrometry", "gradient", "denoise", "color", "stars"]


def setup():
    if ex.w is not None and ex.w.s is not None:
        ex.w.s._dirty = False
    ex.open(ex.project("session"))
    ex.w.activateWindow()
    for b in CHAIN:
        ex.double_click_card(b)
    ex.w.activateWindow()
    ex.settle(4)
    ex.s._undo.clear()


def comp():
    return next((it for it in ex.blocks() if it.node.key == "composition"), None)


setup()
it = comp()
print("comp", it.title, it.node.lid, it.removable, "post", state(ex)["post"])
ex.click_block(it)
it = comp()
print("after click: comp", it and (it.title, it.node.lid), "post", state(ex)["post"], "panel", panel_info(ex)["titles"],
      [b["text"] for b in panel_info(ex)["buttons"]], "undo", ex.s.can_undo())
print("menu", [x.get("text") for x in ex.right_click_block(comp())])
ex.double_click_block(comp())
print("dclick mode", ex.w.mode, EXC[-1:] if EXC else "")
ex.w.set_mode(0)
ex.settle(4)
for action in ("cross", "menu", "del", "drag"):
    setup()
    it = comp()
    n, e = len(TOASTS), len(EXC)
    p0 = state(ex)["post"]
    if action == "cross":
        vp = ex.view.viewport()
        p = ex.view.mapFromScene(it.mapToScene(it.w - 12, 12))
        ex.hover(vp, p)
        ex.click(vp, p)
    elif action == "menu":
        ex.right_click_block(it, pick="Supprimer")
    elif action == "del":
        ex.click_block(it)
        ex.w.activateWindow()
        ex.settle(3)
        ex.key(Qt.Key_Delete)
    else:
        ex.drag_block_to_palette(it)
    ex.settle(4)
    p1 = state(ex)["post"]
    titles = [d["title"] for d in drawn(ex)]
    ex.w.activateWindow()
    ex.settle(3)
    ex.undo()
    p2 = state(ex)["post"]
    titles2 = [d["title"] for d in drawn(ex)]
    print(action, "| post before", len(p0), "after", len(p1), "after undo", len(p2), "| comp drawn",
          "Recombiner" in titles, "after undo", "Recombiner" in titles2,
          "| toasts", TOASTS[n:], "| exc", [x[-150:] for x in EXC[e:]], "| sel", ex.s.sel_step)
ex.close()
