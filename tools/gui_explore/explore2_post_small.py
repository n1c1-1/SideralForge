"""post_wires round 2: at 1366x768, the full chain and the Post block: visible? "Show all"; pulling a port to Post."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtWidgets import QAbstractButton

lang = sys.argv[1] if len(sys.argv) > 1 else "fr"
start(300)
ex = Explorer("post_wires", lang=lang, size=(1366, 768))
pp = P(ex)
session_project(ex)
build_chain(ex, CHAINS["stars"])


def post_visible():
    vr = ex.view.mapToScene(ex.view.viewport().rect()).boundingRect()
    return vr.contains(pp.final().sceneBoundingRect())


print("post visible after building:", post_visible(), "zoom", ex.view.transform().m11())
ex.shot("small_built")
b = [x for x in ex.w.findChildren(QAbstractButton) if x.isVisible() and x.text() in ("Tout afficher", "Show all")]
if b:
    ex.click(b[0])
    ex.settle(6)
print("post visible after Show all:", post_visible(), "zoom", ex.view.transform().m11())
ex.shot("small_show_all")
pp.pull_port(pp.block("etoiles"), "starless")
print("after pull: wires", [w.toolTip().splitlines()[0] for w in pp.wires()], "post", pp.final_summary())
# reopen the project: fitted?
root = ex.s.p.root if hasattr(ex.s.p, "root") else None
ex.close()
