"""post_wires round 2: Fusion to Post (menu, ports, wire removed, Fusion deleted, Ctrl+Z) and the Split block's own
"Vers Post" (which image?)."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start

start(600)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["fusion", "split"]


def st(tag=""):
    return (f"[{tag}] post={pp.final_summary()!r} layers={[n for n, _ in pp.graph_layers()]} "
            f"paths={[q.name for q in ex.s.paths]} wires={[w.toolTip().splitlines()[0] for w in pp.wires()]} "
            f"comps={len(pp.blocks_of('composition'))} toast={pp.toast()!r}")


if "fusion" in which:
    root = ex.project("mono")
    ex.open(root)
    ex.w.activateWindow()
    print(st("mono"))
    ex.double_click_card("fusion")
    ex.settle(8)
    print(st("fusion placed"))
    fu = pp.block("fusion")
    print("fusion tip", fu.toolTip())
    ex.right_click_block(fu, "Vers Post")
    ex.settle(6)
    print(st("fusion Vers Post"))
    ex.shot("fusion_sent")
    ws = [w for w in pp.wires() if "HOO" in w.toolTip()]
    if ws and pp.click_wire(ws[0]):
        pp.del_key()
        print(st("fusion wire removed"))
        pp.undo()
        print(st("undo"))
    pp.pull_port(pp.block("fusion"), "starless")
    print(st("fusion starless port"))
    ex.right_click_block(pp.block("fusion"), "Supprimer")
    ex.settle(6)
    print(st("fusion deleted"))
    ex.shot("fusion_deleted")
    pp.undo()
    print(st("undo"))
    pp.redo()
    print(st("redo"))

if "split" in which:
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    m = ex.right_click_block(pp.block("etoiles"))
    print("split menu tips", [(d.get("text"), d.get("tip")) for d in m if "Post" in (d.get("text") or "")])
    ex.right_click_block(pp.block("etoiles"), "Vers Post")
    ex.settle(6)
    print(st("split Vers Post"))
    ex.click_block(pp.final())
    print("post panel", pp.panel_texts()[2:3])
    ex.shot("split_own_send")
    m = ex.right_click_block(pp.block("etoiles"))
    print("split menu after", [d.get("text") for d in m])
ex.close()
