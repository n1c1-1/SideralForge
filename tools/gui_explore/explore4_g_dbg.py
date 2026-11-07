"""Round 4 (agent G): debug helper, the last block's pull after its wire was removed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402

start(900)
ex = Explorer("g4")
pp = P(ex)
session_project(ex)
build_chain(ex, CHAINS["color"])
if len(sys.argv) > 1:
    pp.pull_port(pp.block("fond"))
ex.page.remove_post_link(ex.s.paths[0].pid)
ex.settle(6)
pp.pull_port(pp.block("couleur"))
print("pulled:", repr(pp.toast()), pp.final_summary())
print("removed:", repr(pp.toast()), pp.final_summary())
g = Graph(ex.s)
q = ex.s.paths[0]
n = g.node_of(q, "couleur")
print("leaf", g.leaves.get(q.pid).key, "node", n.key, g.leaves.get(q.pid) is n, "to_post", g.to_post,
      "gives", ex.s.gives_post(q), "blocks", [b["block"] for b in ex.s.p.workflow(q.pid).blocks])
print("layers", ex.s.post_layers(), "removed", ex.s.post_removed())
ex.close()
