"""Round 4 (agent G): the Fusion card in a colour project (2); the Post page after a language switch (3); a branch's
last own block deleted (4); a wire's panel (6); D on a Fusion and on a channel's Astrometry (15); a variant kept (14).
Prints only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_post_common import CHAINS, P, SLOT_ERRORS, Explorer, build_chain, session_project, start  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

start(900)
ex = Explorer("g4")
pp = P(ex)
what = sys.argv[1] if len(sys.argv) > 1 else "all"
toasts: list = []


def hook():
    ex.page.toasted.connect(toasts.append)


if what in ("all", "2"):
    for tpl in ("classic_dn", "hoo"):
        ex.new_project(f"fc_{tpl}", tpl)
        hook()
        c = ex.card("fusion")
        print("2", tpl, "card reason:", repr(getattr(c, "reason", None)))
        ex.double_click_card("fusion")
        print("2", tpl, "after double-click:", toasts[-1:], [q.name for q in ex.s.paths])

if what in ("all", "3"):
    ex.new_project("lang_lrgb", "lrgb")
    ex.settle(10)
    ex.double_click_card("composition")
    ex.settle(6)
    ex.click(ex.w.mode_btns[2])
    ex.settle(10)
    ex.click(ex.w.mode_btns[0])
    ex.w.change_language("en")
    ex.settle(10)
    ex.double_click_card("denoise")
    pp.undo()
    pp.redo()
    ex.settle(10)
    print("3 slot errors after the switch:", [e.strip().splitlines()[-1] for e in SLOT_ERRORS])
    ex.w.change_language("fr")
    ex.settle(10)

if what in ("all", "4"):
    session_project(ex)
    hook()
    build_chain(ex, CHAINS["astrometry"])
    ex.drag_palette_hover("color", ex.view.viewport().rect().center())
    gh = next((g for g in ex.ghosts() if getattr(g, "branch_key", None) == "astrometrie"), None)
    at = ex.ghost_center(gh)
    ex.drag_leave()
    ex.drag_palette("color", at)
    print("4 branch made:", toasts[-1:], [q.name for q in ex.s.paths])
    br = ex.s.paths[-1]
    ex.click_block(ex.block("couleur", br.pid))
    pp.del_key()
    print("4 after Del:", toasts[-1:], [q.name for q in ex.s.paths], "sel", ex.s.sel_step)
    pp.undo()
    print("4 after Ctrl+Z:", [q.name for q in ex.s.paths])

if what in ("all", "6"):
    session_project(ex)
    hook()
    build_chain(ex, CHAINS["stars"])
    ex.right_click_block(pp.block("etoiles"), pick="Sans étoiles vers Post")
    for w in pp.wires():
        tip = w.toolTip().split("\n")[0]
        if pp.click_wire(w):
            print("6 wire", tip, "-> panel", pp.panel_texts()[:6])
            pp.del_key()
            print("6 Del:", toasts[-1:], pp.final_summary())
            break
    ex.click_block(ex.view.final_item)
    print("11 Post panel:", pp.panel_texts()[:4])

if what in ("all", "15"):
    ex.new_project("d_hoo", "hoo")
    hook()
    ex.click_block(next(it for it in ex.blocks() if it.node.key == "fusion"))
    pp.key(Qt.Key_D)
    print("15 D on Fusion:", toasts[-1:])
    ha = ex.s.paths[0]
    a = ex.block("astrometrie", ha.pid)
    if a is not None:
        ex.click_block(a)
        pp.key(Qt.Key_D)
        print("15 D on Ha Astrometry:", toasts[-1:])
print("errors:", [e.strip().splitlines()[-1] for e in SLOT_ERRORS])
ex.close()
