"""post_wires round 2: frame blocks (Calibration, Alignement, Sélection) never offer "Vers Post"; their ports."""
from explore_post_common import P, Explorer, build_chain, session_project, start

start(300)
ex = Explorer("post_wires")
pp = P(ex)
session_project(ex)
build_chain(ex, ["calibration", "registration", "selection", "stacking"])
for k in ("calibration", "alignement", "selection", "empilement"):
    b = pp.block(k)
    if b is None:
        print(k, "absent", [x.node.key for x in ex.blocks()])
        continue
    m = ex.right_click_block(pp.block(k))
    print(k, "ports", pp.block(k).ports, "menu", [d.get("text") for d in m])
ex.close()
