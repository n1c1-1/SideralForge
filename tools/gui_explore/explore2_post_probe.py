"""post_wires round 2: a first look at the state after the fixes (menus, wires, Post block)."""
from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start

start(300)
ex = Explorer("post_wires")
pp = P(ex)
session_project(ex)
build_chain(ex, CHAINS["stars"])
pp.dump("built stars")
for key in ("empilement", "fond", "couleur", "etoiles"):
    b = pp.block(key)
    print(key, "menu:", [d.get("text") for d in ex.right_click_block(b)] if b else None)
print("post menu:", ex.right_click_block(pp.final()))
for w in pp.wires():
    print("wire menu:", pp.right_click_wire(w))
ex.click_block(pp.final())
print("post panel:", pp.panel_texts(), pp.panel_buttons())
print("cards:", list(ex.page.cards.keys()))
ex.close()
