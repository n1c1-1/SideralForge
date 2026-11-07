"""Probe 1: a session chain built block by block, what goes to Post at each step (no notes, prints only)."""
from explore_post_common import P, Explorer, build_chain, session_project, start

start(300)
ex = Explorer("post_wires_probe")
pp = P(ex)
session_project(ex)
pp.dump("opened")
for b in ["calibration", "registration", "stacking", "astrometry", "gradient", "denoise", "color", "stars"]:
    ex.double_click_card(b)
    ex.settle(8)
    pp.dump(f"after {b}")
    print("  cards:", {k: c.reason for k, c in ex.page.cards.items()})
ex.shot("probe1_full")
for it in ex.blocks():
    print("MENU", it.node.key, [d.get("text") for d in ex.right_click_block(it)])
ex.close()
