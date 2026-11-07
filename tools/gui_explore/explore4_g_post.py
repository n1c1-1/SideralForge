"""Round 4 (agent G): Post wires: a port pulled twice / after its wire was removed (5), the shared Stacking sent after
a Gradient branch (7), the Workflow submenu of Stars/background (9), "Vers Post" place in menus (10), a wire's panel
(6). Prints only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start  # noqa: E402

start(900)
ex = Explorer("g4")
pp = P(ex)
what = sys.argv[1] if len(sys.argv) > 1 else "all"


def names():
    return [q.name for q in ex.s.paths]


if what in ("all", "5"):
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    pp.pull_port(pp.block("fond"))
    print("5 first pull:", repr(pp.toast()), names(), pp.final_summary())
    pp.pull_port(pp.block("fond"))
    print("5 second pull:", repr(pp.toast()), names(), pp.final_summary())
    w = next(w for w in pp.wires() if "Workflow »" in w.toolTip() or "“Workflow”" in w.toolTip()
             or w.toolTip().startswith("« Workflow »"))
    pp.click_wire(w)
    print("6 wire panel:", pp.panel_texts()[:8])
    pp.del_key()
    print("5 wire removed:", repr(pp.toast()), names(), pp.final_summary())
    pp.pull_port(pp.block("couleur"))
    print("5 pull last after wire removed:", repr(pp.toast()), names(), pp.final_summary())
    pp.check_slot_errors("5")

if what in ("all", "7"):
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), pick="Vers Post")
    print("7 gradient sent:", repr(pp.toast()), names())
    pp.pull_port(pp.block("empilement"))
    print("7 stacking pulled:", repr(pp.toast()), names())
    ex.right_click_block(pp.block("empilement"), pick="Vers Post")
    print("7 stacking menu again:", repr(pp.toast()), names())

if what in ("all", "9", "10"):
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    for k in ("empilement", "fond", "debruitage", "couleur", "etoiles"):
        m = ex.right_click_block(pp.block(k))
        print("10", k, [d["text"] for d in m], [[x["text"] for x in d.get("sub") or []] for d in m if d.get("sub")])
pp.check_slot_errors("end")
ex.close()
