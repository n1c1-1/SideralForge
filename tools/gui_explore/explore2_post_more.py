"""post_wires round 2: a workflow taken out of Post then grown by a block; a Session deleted with send branches, Ctrl+Z;
Combine re-added after all removed; the Split's ports after the starless sent and Recombine deleted."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start

start(600)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["grow", "session", "recomb"]


def st(tag=""):
    return (f"[{tag}] post={pp.final_summary()!r} layers={pp.graph_layers()} "
            f"paths={[(q.name, [n['block'] for n in (ex.s.p.workflow(q.pid).blocks or [])][-2:]) for q in ex.s.paths]} "
            f"wires={[w.toolTip().splitlines()[0] for w in pp.wires()]} toast={pp.toast()!r}")


if "grow" in which:
    session_project(ex)
    build_chain(ex, CHAINS["gradient"])
    pp.click_wire(pp.wires()[0])
    pp.del_key()
    print(st("wire removed"))
    ex.double_click_card("denoise")
    ex.settle(6)
    print(st("denoise added at the end"))
    ex.shot("grow_after_removed")
    ex.double_click_card("color")
    ex.settle(6)
    ex.double_click_card("stars")
    ex.settle(6)
    print(st("color + stars added"), [(b.title) for b in pp.blocks_of("composition")])

if "session" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    ex.right_click_block(pp.block("empilement", pid=ex.s.paths[0].pid), "Vers Post")
    ex.settle(6)
    print(st("2 branches"))
    src = pp.block("sources") or pp.block("session")
    m = ex.right_click_block(src)
    print("session menu", [d.get("text") for d in m])
    ex.right_click_block(src, "Supprimer")
    ex.settle(6)
    print(st("session deleted"), ex.popups[-1:])
    ex.shot("session_deleted_branches")
    pp.undo()
    print(st("undo"))
    pp.redo()
    print(st("redo"))

if "recomb" in which:
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    ex.click_block(pp.blocks_of("composition")[0])
    pp.del_key()
    print(st("recombine deleted"))
    ex.click_block(pp.final())
    print("post panel", pp.panel_texts())
    ex.shot("recomb_none")
    m = ex.right_click_block(pp.block("etoiles"))
    print("split menu", [d.get("text") for d in m])
    ex.right_click_block(pp.block("etoiles"), "+ Recombiner")
    ex.settle(6)
    print(st("+ Recombiner"))
    pp.undo()
    print(st("undo"))
ex.close()
