"""post_wires round 2: the Post page with layers (simulated computed state): 1366x768 and English screenshots, a layer
duplicated / renamed in Post and what the graph says."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit

lang = sys.argv[1] if len(sys.argv) > 1 else "fr"
size = tuple(int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1366x768").split("x"))
start(600)
ex = Explorer("post_wires", lang=lang, size=size)
pp = P(ex)
tag = f"{lang}_{size[0]}"


def simulate():
    s = ex.s
    orig = s._sync

    def fake(*a, **k):
        orig(*a, **k)
        for v in s.paths:
            v.rendered = True
            v.computed = False
    s._sync = fake
    s._sync()


session_project(ex)
build_chain(ex, CHAINS["stars"])
simulate()
pp.pull_port(pp.block("etoiles"), "starless") if size[0] > 1500 else None
ex.right_click_block(pp.block("fond"), "Vers Post" if lang == "fr" else "To Post")
ex.settle(6)
ex.click(ex.w.mode_btns[2])
ex.settle(20)
v = ex.w.post_holder.page.view
print("rows", [r.n.text() for r in v.rows])
lay = next(lay for lay in v.layers if lay["kind"] == "workflow")
v.select(lay["id"], "layer")
ex.settle(4)
ex.shot(f"page2_{tag}")
# duplicate the layer
pp.key(Qt.Key_J, v, Qt.ControlModifier)
ex.settle(6)
print("after Ctrl+J rows", [r.n.text() for r in v.rows])
ex.click(ex.w.mode_btns[0])
ex.settle(8)
ex.click_block(pp.final())
print("graph post after duplicate:", pp.final_summary(), pp.panel_texts()[2:3])
ex.click(ex.w.mode_btns[2])
ex.settle(10)
# rename a workflow layer in Post (double-click on its name, as the user)
row = next(r for r in v.rows if r.lay["kind"] == "workflow" and "Gradient" in r.lay["name"] or "output" in r.lay["name"])


def h(d):
    e = d.findChildren(QLineEdit)
    if e:
        e[0].setText("Fond seul")
        d.accept()
    else:
        d.reject()


ex.dialog_handler = h
ex.double_click(row.n) if hasattr(row, "n") else None
ex.settle(8)
eds = [e for e in v.findChildren(QLineEdit) if e.isVisible()]
if eds:
    eds[0].setText("Fond seul")
    from PySide6.QtTest import QTest
    QTest.keyClick(eds[0], Qt.Key_Return)
    ex.settle(6)
print("after rename in Post rows", [r.n.text() for r in v.rows], "modals", ex.modals[-1:])
ex.click(ex.w.mode_btns[0])
ex.settle(8)
ex.click_block(pp.final())
print("graph after rename in Post: paths", [q.name for q in ex.s.paths], "post panel", pp.panel_texts()[2:3],
      "outs", [(b.summary) for b in pp.blocks_of("post")])
ex.shot(f"page2_{tag}_graph_after_rename")
ex.close()
