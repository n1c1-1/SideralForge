"""post_wires round 2: layer names when the same kind of block is sent from two flows / two channels; the same block
sent twice."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start

start(600)
ex = Explorer("post_wires", lang=(sys.argv[1] if len(sys.argv) > 1 else "fr"))
pp = P(ex)


def paths():
    return [(q.name, [n['block'] for n in (ex.s.p.workflow(q.pid).blocks or [])]) for q in ex.s.paths]


def st(tag=""):
    return (f"[{tag}] post={pp.final_summary()!r} layers={pp.graph_layers()} "
            f"wires={[w.toolTip().splitlines()[0] for w in pp.wires()]} toast={pp.toast()!r}")


# two flows: Empilement of each sent
root = ex.project("two_flows")
ex.open(root)
ex.w.activateWindow()
for pid in [q.pid for q in ex.s.paths]:
    for b in CHAINS["gradient"]:
        ex.page.place_block(b, pid)
        ex.settle(4)
p1, p2 = ex.s.paths[0].pid, ex.s.paths[1].pid
ex.right_click_block(pp.block("empilement", pid=p1), "Vers Post")
ex.settle(6)
ex.right_click_block(pp.block("empilement", pid=p2), "Vers Post")
ex.settle(6)
print(st("two flows, both stackings sent"), paths())
ex.click_block(pp.final())
print("post panel", pp.panel_texts()[2:3])
ex.shot("names_two_flows")
# the same block sent twice (menu), then port
n = len(ex.s.paths)
ex.right_click_block(pp.block("empilement", pid=p1), "Vers Post")
ex.settle(6)
print(st("flow1 stacking sent again (menu)"), len(ex.s.paths) - n)
m = ex.right_click_block(pp.block("empilement", pid=p1))
print("menu after sent", [d.get("text") for d in m])
pp.pull_port(pp.block("empilement", pid=p1))
print(st("again by port"), len(ex.s.paths) - n)
# mono: both channels' Débruitage sent
root = ex.project("mono")
ex.open(root)
ex.w.activateWindow()
ex.double_click_card("fusion")
ex.settle(8)
for f in ("Ha", "OIII"):
    q = next(q for q in ex.s.paths if q.name == f)
    ex.right_click_block(pp.block("debruitage", pid=q.pid), "Vers Post")
    ex.settle(6)
print(st("Ha and OIII denoise sent"), paths())
ex.click_block(pp.final())
print("post panel", pp.panel_texts()[2:3])
ex.shot("names_mono_channels")
ex.close()
