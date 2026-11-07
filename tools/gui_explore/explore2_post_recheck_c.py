"""post_wires round 2, recheck C: Split stars / Combine (#16 #18 #19 #20-#24 #30 #38-#41), #12 (Delete workflow on the
Output block), #9 redo."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtWidgets import QAbstractButton

start(900)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["out12", "ports", "combine", "split_del", "rename", "panel"]


def R(n, *a):
    print(f"R{n}:", *a, flush=True)


def paths():
    return [(q.name, [n['block'] for n in ex.s.p.workflow(q.pid).blocks]) for q in ex.s.paths]


def st(tag=""):
    comps = pp.blocks_of("composition")
    return (f"[{tag}] post={pp.final_summary()!r} layers={pp.graph_layers()} comps={[c.title for c in comps]} "
            f"wires={[w.toolTip().splitlines()[0] for w in pp.wires()]} split={pp.block('etoiles') is not None} "
            f"toast={pp.toast()!r}")


def fresh():
    session_project(ex)
    build_chain(ex, CHAINS["stars"])


if "out12" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    n, nm = len(ex.popups), len(ex.modals)
    ex.right_click_block(pp.blocks_of("post")[0], "Supprimer workflow")
    ex.settle(6)
    R(12, "popups", ex.popups[n:], "modals", ex.modals[nm:], "toast", pp.toast(), "paths", paths())
    ex.shot("recheck_12_delete_workflow")
    pp.undo()
    R(12, "undo", paths(), pp.final_summary())
    pp.redo()
    R(12, "redo", paths(), pp.final_summary())
    pp.undo()
    ex.click_block(pp.blocks_of("post")[0])
    pp.del_key()
    R(9, "del toast", pp.toast(), paths())
    pp.undo()
    R(9, "undo after refused del", paths())
    pp.redo()
    R(9, "redo", paths())

if "ports" in which:
    fresh()
    R(0, st("start"))
    pp.pull_port(pp.block("etoiles"), "starless")
    R(30, st("starless pulled"))
    ex.right_click_block(pp.block("couleur"), "Vers Post")
    ex.settle(6)
    R(30, st("couleur sent after starless"), paths())
    ex.shot("recheck_30_split_then_color")
    pp.undo()
    R(30, st("undo couleur send"), paths())
    pp.pull_port(pp.block("etoiles"), "stars")
    R(16, st("stars pulled"))
    w = next((x for x in pp.wires() if "toiles seules" in x.toolTip() or "Étoiles →" in x.toolTip()), None)
    R(16, "stars wire", w.toolTip() if w else None)
    if w is not None and pp.click_wire(w):
        pp.del_key()
        R(16, st("Del on stars wire"))
        pp.undo()
        R(16, st("undo"))
        ex.shot("recheck_16_undo_stars_wire")
        pp.redo()
        R(16, st("redo"))
        pp.undo()
    # undo until Split goes
    k = 0
    while pp.block("etoiles") is not None and ex.s.can_undo() and k < 10:
        pp.undo()
        k += 1
    R(18, st(f"after {k} undo, split gone"))
    ex.click_block(pp.final())
    R(18, "post panel", pp.panel_texts())
    ex.shot("recheck_18_split_undone")
    m = ex.right_click_block(pp.block("couleur") or pp.final())
    R(19, "menu", [d.get("text") for d in m])

if "combine" in which:
    fresh()
    R(20, st("start"))
    cb = pp.blocks_of("composition")[0]
    ex.click_block(cb)
    R(41, "combine panel", pp.panel_texts(), pp.panel_buttons())
    pp.del_key()
    R(20, st("Del combine"))
    R(41, "toast after Del", pp.toast())
    pp.undo()
    R(20, st("undo"))
    ex.shot("recheck_20_undo_combine")
    pp.redo()
    R(20, st("redo"))
    pp.undo()
    ex.double_click_card("composition")
    ex.settle(6)
    ex.right_click_block(pp.block("etoiles"), "+ Recombiner")
    ex.settle(6)
    R(21, st("3 combines"), [(c.title, c.summary) for c in pp.blocks_of("composition")])
    ex.shot("recheck_21_combine_names")
    ex.click_block(pp.final())
    R(21, "post panel", pp.panel_texts())
    cb = pp.blocks_of("composition")[0]
    ex.hover(ex.view.viewport(), ex.point_of(cb))
    ex.click(ex.view.viewport(), ex.point_of(cb, "cross"))
    R(23, st("cross on a combine"))
    pp.undo()
    R(23, st("undo after cross"))
    ex.shot("recheck_23_undo_cross")
    # panel button delete
    cb = pp.blocks_of("composition")[-1]
    ex.click_block(cb)
    b = [x for x in ex.page.panel.findChildren(QAbstractButton) if x.isVisible() and "Supprimer" in x.text()]
    if b:
        ex.click(b[0])
        R(41, st("panel Supprimer"))
        ex.shot("recheck_41_panel_delete")
        pp.undo()
        R(41, st("undo"))

if "split_del" in which:
    fresh()
    ex.double_click_card("composition")
    ex.settle(6)
    R(24, st("2 combines"))
    ex.right_click_block(pp.block("etoiles"), "Supprimer")
    ex.settle(6)
    R(24, st("split deleted"), paths())
    ex.shot("recheck_24_split_deleted")
    pp.undo()
    R(24, st("undo"))
    pp.redo()
    R(24, st("redo"))

if "rename" in which:
    fresh()
    pp.pull_port(pp.block("etoiles"), "starless")
    ex.dialog_handler = None
    from PySide6.QtWidgets import QInputDialog
    QInputDialog.getText = staticmethod(lambda *a, **k: ("M31 final", True))
    ex.right_click_block(pp.block("etoiles"), "Renommer")
    ex.settle(6)
    R(38, "paths", [q.name for q in ex.s.paths], "modals", ex.modals[-2:])
    ex.click_block(pp.final())
    R(38, st("renamed"), pp.panel_texts())
    ex.shot("recheck_38_renamed")
    pp.undo()
    R(38, st("undo rename"), [q.name for q in ex.s.paths])
    pp.redo()
    R(38, st("redo rename"), [q.name for q in ex.s.paths])

if "panel" in which:
    fresh()
    ex.click_block(pp.block("etoiles"))
    R(40, "split panel", pp.panel_buttons())
    b = [x for x in ex.page.panel.findChildren(QAbstractButton) if x.isVisible() and "Post" in x.text()]
    R(40, "post buttons", [x.text() for x in b])
    if b:
        ex.click(b[0])
        R(40, st("panel send"))
        ex.shot("recheck_40_panel_send")
        pp.undo()
        R(40, st("undo panel send"))
        pp.redo()
        ex.click_block(pp.block("etoiles"))
        b = [x for x in ex.page.panel.findChildren(QAbstractButton) if x.isVisible() and "Post" in x.text()]
        R(39, "post buttons after send", [x.text() for x in b])
        out = [x for x in b if "Retirer" in x.text() or "hors" in x.text().lower()]
        if out:
            ex.click(out[0])
            R(39, st("panel take out"))
            ex.shot("recheck_39_panel_out")
            pp.undo()
            R(39, st("undo take out"))
print("popups", ex.popups, "modals", ex.modals, "slot", pp.take_slot_errors())
ex.close()
