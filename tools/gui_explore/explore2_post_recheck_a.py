"""post_wires round 2, recheck A: the wire of a chain to Post, images of middle blocks sent, branches (round-1 #1-#15,
#25, #26, #29, #31, #32). Prints facts; screenshots recheck_*.png."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtWidgets import QApplication

start(900)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["page", "send", "wire", "branch", "lost"]


def R(n, *a):
    print(f"R{n}:", *a, flush=True)


def paths():
    return [(q.name, [n['block'] for n in ex.s.p.workflow(q.pid).blocks]) for q in ex.s.paths]


if "page" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.click_block(pp.final())
    R(1, "post panel buttons", pp.panel_buttons())
    R(2, "post menu", ex.right_click_block(pp.final()))
    pp.dump("color chain")
    ex.double_click_block(pp.final())
    ex.settle(8)
    R(3, "after double-click Post: mode", ex.w.mode,
      [t[:120] for t in ex.texts(ex.w.post) if t.strip()][:25] if ex.w.post is not None else None)
    ex.shot("recheck_03_post_page")
    ex.w.set_mode(0)
    ex.settle(4)

if "send" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    pp.dump("gradient sent")
    R(5, "blocks", pp.blocks_desc())
    R(25, "toast", pp.toast())
    ex.shot("recheck_05_gradient_sent")
    outs = pp.blocks_of("output")
    R(7, "output blocks", [(o.title) for o in outs])
    if outs:
        R(7, "output menu", [d.get("text") for d in ex.right_click_block(outs[0])])
    ex.click_block(pp.final())
    R(10, "post panel", pp.panel_texts())
    # Delete workflow of the branch: confirmation?
    n = len(ex.popups)
    br = ex.s.paths[-1]
    last = [b for b in ex.blocks() if any(q.pid == br.pid for q in b.node.paths) and len(b.node.paths) == 1]
    R(12, "branch own blocks", [(b.node.key, b.title) for b in last])
    tail = [b for b in ex.blocks() if any(q.pid == br.pid for q in b.node.paths)]
    tail = sorted(tail, key=lambda b: b.node.col)[-1]
    m = ex.right_click_block(tail)
    R(12, "menu of branch's last block", tail.title, [d.get("text") for d in m])
    if any("workflow" in (d.get("text") or "") for d in m):
        ex.right_click_block(tail, "Supprimer workflow")
        ex.settle(6)
        R(12, "popups", ex.popups[n:], "toast", pp.toast(), "paths", paths())
        pp.undo()
        R(12, "after undo paths", paths())
    # stacking sent: elision
    ex.right_click_block(pp.block("empilement", pid=ex.s.paths[0].pid), "Vers Post")
    ex.settle(6)
    R(25, "toast stacking", pp.toast(), [q.name for q in ex.s.paths])
    ex.right_click_block(pp.block("astrometrie", pid=ex.s.paths[0].pid), "Vers Post")
    ex.settle(6)
    R(25, "toast astrometry", pp.toast(), [q.name for q in ex.s.paths])
    ex.shot("recheck_25_toast")

if "wire" in which:
    session_project(ex)
    build_chain(ex, CHAINS["stacking"])
    ws = pp.wires()
    R(4, "wires", [w.toolTip() for w in ws])
    if ws:
        pp.click_wire(ws[0])
        R(4, "selected blocks", [b.title for b in ex.blocks() if b.selected], "sel_step", ex.s.sel_step,
          "panel", pp.panel_texts()[:3], "wire sel", ws[0].selected)
        ex.shot("recheck_04_wire_selected")
        pp.del_key()
        R(8, "toast", pp.toast(), "wires", len(pp.wires()), "post", pp.final_summary())
        ex.shot("recheck_08_toast")
        before = paths()
        pp.undo()
        R(11, "undo: wires", len(pp.wires()), "paths", before, "->", paths(), "post", pp.final_summary())
        ex.shot("recheck_11_undo")
        pp.redo()
        R(11, "redo: wires", len(pp.wires()), "post", pp.final_summary(), "toast", pp.toast())
        m = ex.right_click_block(pp.block("empilement"))
        R(13, "menu after removal", [d.get("text") for d in m])
        ex.right_click_block(pp.block("empilement"), "Vers Post")
        ex.settle(6)
        R(13, "sent back: wires", len(pp.wires()), "paths", paths(), "post", pp.final_summary(), "toast", pp.toast())
        pp.undo()
        R(13, "undo send back: wires", len(pp.wires()), "paths", paths(), "post", pp.final_summary())
        ex.shot("recheck_13_undo_sendback")
        pp.redo()
        R(13, "redo: wires", len(pp.wires()), "post", pp.final_summary())
        ws = pp.wires()
        if ws:
            m = pp.right_click_wire(ws[0], "Retirer")
            pp.check_slot_errors("wire menu remove")
            R(14, "wire menu", m, "wires after", len(pp.wires()), "toast", pp.toast(), "slot errors", pp.take_slot_errors())
            ex.shot("recheck_15_wire_menu")

if "branch" in which:
    # #26 send stacking, remove its wire: branch stays?
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("empilement"), "Vers Post")
    ex.settle(6)
    br = ex.s.paths[-1]
    R(26, "after send", paths(), [w.toolTip().splitlines()[0] for w in pp.wires()], pp.final_summary())
    wl = [w for w in pp.wires() if br.name in w.toolTip()]
    if wl and pp.click_wire(wl[0]):
        pp.del_key()
        R(26, "after Del on its wire", paths(), pp.final_summary(), "toast", pp.toast())
        ex.shot("recheck_26_branch_wire_removed")
        pp.undo()
        R(26, "undo", paths(), pp.final_summary(), len(pp.wires()))
        pp.redo()
        R(26, "redo", paths(), pp.final_summary(), len(pp.wires()))
    else:
        R(26, "no wire for branch", [w.toolTip() for w in pp.wires()])
    # #31 message of branch
    session_project(ex)
    build_chain(ex, CHAINS["gradient"])
    ex.drag_palette_hover("denoise", ex.view.viewport().rect().center())
    gh = [(g.text, ex.ghost_center(g)) for g in ex.ghosts()]
    ex.drag_leave()
    R(31, "ghosts", [t for t, _ in gh])
    br = next((c for t, c in gh if "branche" in t.lower()), None)
    if br is not None:
        ex.drag_palette("denoise", br)
        R(31, "toast", pp.toast(), paths(), pp.final_summary())
        ex.shot("recheck_31_branch_toast")

if "lost" in which:
    # #29 after wire removal, send from middle block
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    pp.click_wire(pp.wires()[0])
    pp.del_key()
    ex.right_click_block(pp.block("couleur"), "Vers Post")
    ex.settle(6)
    ex.right_click_block(pp.block("fond", pid=ex.s.paths[0].pid), "Vers Post")
    ex.settle(6)
    R(29, "paths", paths(), "post", pp.final_summary(), "layers", pp.graph_layers(), "wires",
      [w.toolTip().splitlines()[0] for w in pp.wires()], "toast", pp.toast())
    ex.shot("recheck_29_after_removal_send")
    # #32 new branch after composition changed
    session_project(ex)
    build_chain(ex, CHAINS["gradient"])
    pp.click_wire(pp.wires()[0])
    pp.del_key()
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    ex.drag_palette_hover("denoise", ex.view.viewport().rect().center())
    gh = [(g.text, ex.ghost_center(g)) for g in ex.ghosts()]
    ex.drag_leave()
    br = next((c for t, c in gh if "branche" in t.lower() and "Empilement" in t), None) or \
        next((c for t, c in gh if "branche" in t.lower()), None)
    R(32, "ghosts", [t for t, _ in gh])
    if br is not None:
        ex.drag_palette("denoise", br)
        R(32, "paths", paths(), "post", pp.final_summary(), "layers", pp.graph_layers(), "wires",
          [w.toolTip().splitlines()[0] for w in pp.wires()], "toast", pp.toast())
        ex.shot("recheck_32_new_branch")
print("popups", ex.popups, "modals", ex.modals, "slot", pp.take_slot_errors())
ex.close()
