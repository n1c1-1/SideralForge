"""post_wires round 2: are the layers sent to Post shown (eye/checkbox on) when they arrive? Twice, from fresh projects;
and a Post layer renamed: the graph's workflow name."""
from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit

start(600)
ex = Explorer("post_wires")
pp = P(ex)


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


for run in (1, 2):
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    simulate()
    pp.pull_port(pp.block("etoiles"), "starless")
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    pp.pull_port(pp.block("empilement", pid=ex.s.paths[0].pid))
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    v = ex.w.post_holder.page.view
    print(run, "layers", [(lay["name"], lay.get("visible")) for lay in v.layers])
    ex.shot(f"visible_run{run}")
    ex.click(ex.w.mode_btns[0])
    ex.settle(6)
    # send after Post was opened once
    ex.right_click_block(pp.block("couleur"), "Vers Post")
    ex.settle(6)
    ex.click(ex.w.mode_btns[2])
    ex.settle(15)
    print(run, "after Couleur sent with Post opened", [(lay["name"], lay.get("visible")) for lay in v.layers])
    # rename the "Sortie Gradient" layer in Post
    row = next(r for r in v.rows if r.lay["name"] == "Sortie Gradient")

    def h(d):
        e = d.findChildren(QLineEdit)
        e[0].setText("Fond seul")
        d.accept()
    ex.dialog_handler = h
    ex.double_click(row.n)
    ex.settle(8)
    print(run, 'rows after rename', [r.n.text() for r in v.rows], ex.modals[-1:])
    ex.click(ex.w.mode_btns[0])
    ex.settle(8)
    print(run, "layer renamed in Post: workflows", [q.name for q in ex.s.paths],
          "outputs", [b.summary for b in pp.blocks_of("post")], "wires", [w.toolTip().splitlines()[0] for w in pp.wires()])
    ex.click_block(pp.final())
    print(run, 'post panel', pp.panel_texts()[2][:400])
    ex.shot(f"visible_rename_run{run}")
ex.close()
