"""post_wires round 2: whose Ctrl+Z (graph vs Post page); the Output block's name field."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QLineEdit

start(600)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["scopes", "name"]


def st(tag=""):
    return (f"[{tag}] post={pp.final_summary()!r} paths={[q.name for q in ex.s.paths]} "
            f"wires={len(pp.wires())}")


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


if "scopes" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    print(st("sent"))
    # graph edit then Post page, Ctrl+Z there
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    v = ex.w.post_holder.page.view
    print("in Post: rows", [r.n.text() for r in v.rows], "post undo", len(v.undo))
    pp.undo()
    ex.settle(6)
    print("Ctrl+Z in Post with no Post edit:", st(), "rows", [r.n.text() for r in v.rows])
    # a Post edit, back to the graph, Ctrl+Z
    lay = v.layers[0]
    v.select(lay["id"], "layer")
    ex.settle(4)
    c = next(c for c in v.right.findChildren(QComboBox) if c.isVisible() and c.count() > 2)
    c.setCurrentIndex(2)
    ex.settle(6)
    ex.click(ex.w.mode_btns[0])
    ex.settle(8)
    pp.undo()
    print("Ctrl+Z in graph after a Post edit:", st(), "post modes", [(x["name"], x.get("mode")) for x in v.layers])
    ex.shot("scopes_graph_undo")
    # Édition menu: what does it say?
    mb = ex.w.menuBar()
    for a in mb.actions():
        if a.text().replace("&", "") in ("Édition", "Edit"):
            print("edit menu:", [(x.text(), x.isEnabled()) for x in a.menu().actions()])

if "name" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    ex.click_block(pp.blocks_of("post")[0])
    eds = [e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisible()]
    print("name fields", [(e.text(), e.placeholderText(), e.toolTip()) for e in eds])
    if eds:
        e = eds[0]
        ex.click(e)
        e.selectAll()
        QTest.keyClicks(e, "Fond net")
        QTest.keyClick(e, Qt.Key_Return)
        ex.settle(8)
        print(st("renamed in panel"), [(b.title, b.summary) for b in pp.blocks_of("post")], pp.graph_layers())
        pp.undo()
        print(st("undo"), pp.graph_layers())
        pp.redo()
        print(st("redo"), pp.graph_layers())
        # empty name
        ex.click_block(pp.blocks_of("post")[0])
        e = [e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisible()][0]
        e.selectAll()
        QTest.keyClick(e, Qt.Key_Delete)
        QTest.keyClick(e, Qt.Key_Return)
        ex.settle(8)
        print(st("empty name"), pp.graph_layers(), repr(pp.toast()))
        ex.shot("name_empty")
ex.close()
