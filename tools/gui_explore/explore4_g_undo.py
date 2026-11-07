"""Round 4 (agent G): selection after Ctrl+Z of a placed block (12); Session 2 joining message (16); a raw file taken
out then Ctrl+Z (13). Prints only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_post_common import P, Explorer, start  # noqa: E402

start(900)
ex = Explorer("g4")
pp = P(ex)
what = sys.argv[1] if len(sys.argv) > 1 else "all"
toasts = []
if what in ("all", "12"):
    ex.open(ex.project("session_empty_wf"))
    ex.page.toasted.connect(toasts.append)
    for b in ("calibration", "registration", "stacking"):
        ex.double_click_card(b)
    print("12 before undo:", repr(ex.s.sel_step), [q.name for q in ex.s.paths])
    pp.undo()
    ex.settle(10)
    print("12 after undo:", repr(ex.s.sel_step), "panel:", pp.panel_texts()[:3], "selected items:",
          [it.title for it in ex.blocks() if it.selected])
    pp.redo()
    ex.settle(10)
    print("12 after redo:", repr(ex.s.sel_step), [it.title for it in ex.blocks() if it.selected])
if what in ("all", "16"):
    ex.open(ex.project("session_empty_wf"))
    ex.page.toasted.connect(toasts.append)
    for b in ("calibration", "registration", "stacking"):
        ex.double_click_card(b)
    ex.double_click_card("session")
    ex.settle(20)
    print("16 toast shown:", repr(pp.toast()), "| last toasts:", toasts[-3:])
ex.close()
