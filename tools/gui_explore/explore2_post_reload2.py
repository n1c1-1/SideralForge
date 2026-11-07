"""post_wires round 2: the main workflow's layer deleted in the Post page: the graph's wire; after Save and reopen."""
from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt

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


def st(tag):
    return (f"[{tag}] post={pp.final_summary()!r} layers={[n for n, _ in pp.graph_layers()]} "
            f"paths={[q.name for q in ex.s.paths]} wires={[w.toolTip().splitlines()[0] for w in pp.wires()]}")


root = session_project(ex)
build_chain(ex, CHAINS["color"])
simulate()
ex.click(ex.w.mode_btns[2])
ex.settle(20)
v = ex.w.post_holder.page.view
lay = next(lay for lay in v.layers if lay["name"] == "Workflow")
v.select(lay["id"], "layer")
ex.settle(3)
print('del button', v.b_del.isEnabled(), repr(v.b_del.toolTip()), 'layers before', [l['name'] for l in v.layers])
ex.click(v.b_del)
print('popups', ex.popups[-2:], 'toast/status', [t for t in ex.texts(ex.w.post) if 'calque' in t.lower()][:5])
ex.settle(10)
print("post rows after delete", [r.n.text() for r in v.rows])
ex.click(ex.w.mode_btns[0])
ex.settle(10)
print(st("graph after main layer deleted in Post"))
ex.shot("reload2_main_layer_deleted")
pp.key(Qt.Key_S, ex.w, Qt.ControlModifier)
ex.settle(8)
ex.w.open_project(root)
ex.settle(12)
print(st("reopened"))
ex.click(ex.w.mode_btns[2])
ex.settle(15)
v = ex.w.post_holder.page.view
print("post rows after reopen", [r.n.text() for r in v.rows])
ex.close()
