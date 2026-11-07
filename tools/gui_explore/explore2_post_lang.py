"""post_wires round 2: the same gestures in English (or French) at a given screen size: what the user reads (menus,
messages, panels, the Post page), screenshots."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt

lang = sys.argv[1] if len(sys.argv) > 1 else "en"
size = tuple(int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1366x768").split("x"))
start(600)
ex = Explorer("post_wires", lang=lang, size=size)
pp = P(ex)
tag = f"{lang}_{size[0]}"

session_project(ex)
build_chain(ex, CHAINS["stars"])
print("blocks", [(b.title, b.summary) for b in ex.blocks()])
ex.click_block(pp.final())
print("post panel", pp.panel_texts(), pp.panel_buttons())
print("post block tip", pp.final().toolTip())
ex.shot(f"lang_{tag}_post_panel")
for key in ("fond", "etoiles"):
    print(key, "menu", [d.get("text") for d in ex.right_click_block(pp.block(key))])
print("combine menu", [d.get("text") for d in ex.right_click_block(pp.blocks_of("composition")[0])])
ex.click_block(pp.block("etoiles"))
print("split panel", pp.panel_texts()[:14])
ex.shot(f"lang_{tag}_split_panel")
ex.right_click_block(pp.block("fond"), "Vers Post" if lang == "fr" else "To Post")
ex.settle(6)
print("toast send", repr(pp.toast()), [q.name for q in ex.s.paths])
outb = pp.blocks_of("post")
print("output blocks", [(b.title, b.summary, b.toolTip()) for b in outb])
ex.shot(f"lang_{tag}_sent")
if outb:
    ex.click_block(outb[0])
    print("output panel", pp.panel_texts())
    print("output menu", [d.get("text") for d in ex.right_click_block(pp.blocks_of("post")[0])])
    ex.click_block(pp.blocks_of("post")[0])
    pp.del_key()
    print("del output toast", repr(pp.toast()))
pp.pull_port(pp.block("etoiles"), "starless")
print("toast starless", repr(pp.toast()))
ws = pp.wires()
print("wires", [w.toolTip() for w in ws])
w = next((w for w in ws if "Gradient" in w.toolTip()), None)
if w is not None:
    print("wire menu", pp.right_click_wire(w))
    pp.click_wire(w)
    pp.del_key()
    print("toast wire del", repr(pp.toast()))
    pp.undo()
    print("toast undo", repr(pp.toast()), "status", ex.w.statusBar().currentMessage() if ex.w.statusBar() else None)
ex.click_block(pp.final())
print("post panel 2", pp.panel_texts())
ex.shot(f"lang_{tag}_graph_after")
ex.double_click_block(pp.final())
ex.settle(8)
print("post page texts", [t for t in ex.texts(ex.w.post) if t.strip()][:6])
ex.shot(f"lang_{tag}_post_page")
ex.close()
