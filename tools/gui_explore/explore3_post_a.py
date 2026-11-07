"""post_wires round 3 (control): replay the round-2 points with real gestures and print what is seen. Args: the
scenarios to run (default all); each is run twice (fresh project each time). Nothing is noted here."""
import sys
import time

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QComboBox, QLineEdit

start(1500)
ex = Explorer("post_wires")
pp = P(ex)
RUN = [1]


def wait(sec):
    t = time.time()
    while time.time() - t < sec:
        ex.settle(5)


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


def shot(name):
    return ex.shot(f"{name}_run{RUN[0]}")


def wires():
    return [w.toolTip().splitlines()[0] for w in pp.wires()]


def state():
    return {"paths": [q.name for q in ex.s.paths], "blocks": sorted(it.node.key for it in ex.blocks()),
            "wires": wires(), "final": pp.final_summary(), "layers": pp.graph_layers(), "toast": pp.toast()}


def wire_of(part):
    return next((w for w in pp.wires() if part in w.toolTip()), None)


def rename_to(name):
    def h(d):
        e = d.findChildren(QLineEdit)
        if e:
            e[0].setText(name)
            d.accept()
        else:
            d.reject()
    return h


def post_page():
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    return ex.w.post_holder.page.view


def graph_page():
    ex.click(ex.w.mode_btns[0])
    ex.settle(8)


# ---------------------------------------------------------------- new anomalies of round 2
def n1():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    v = post_page()
    print(" layers in Post", [x["name"] for x in v.layers])
    tgt = next(lay for lay in v.layers if "Gradient" in lay["name"])
    v.select(tgt["id"], "layer")
    ex.settle(3)
    print(" b_del enabled", v.b_del.isEnabled(), v.b_del.toolTip())
    ex.click(v.b_del)
    ex.settle(10)
    print(" after delete in Post", [x["name"] for x in v.layers])
    graph_page()
    print(" graph", state())
    shot("n1_layer_deleted_in_post")


def n2():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    v = post_page()
    lay = next(lay for lay in v.layers if "Gradient" in lay["name"])
    v.select(lay["id"], "layer")
    ex.settle(4)
    c = next(c for c in v.right.findChildren(QComboBox) if c.isVisible() and c.count() > 2)
    c.setCurrentIndex(2)
    ex.settle(6)
    before = [(x["name"], x.get("mode")) for x in v.layers]
    graph_page()
    w = wire_of("Gradient")
    print(" wire", w.toolTip() if w else None)
    pp.click_wire(w)
    pp.del_key()
    print(" after del", state())
    pp.undo()
    print(" after undo", state())
    v = post_page()
    after = [(x["name"], x.get("mode")) for x in v.layers]
    print(" before", before, "after", after)
    shot("n2_layer_settings")


def n3():
    ex.new_project(f"DeuxImages{time.time():.0f}")
    ex.w.activateWindow()
    ex.double_click_card("image")
    ex.settle(8)
    print(" after 1st", state(), pp.toast())
    ex.double_click_card("image")
    ex.settle(8)
    print(" after 2nd", state(), "images", len(pp.blocks_of("image")))
    shot("n3_two_images")


def n4():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    print(" sent", state())
    w = wire_of("Gradient")
    pp.click_wire(w)
    print(" panel after wire click", pp.panel_texts()[:4])
    ex.double_click_card("denoise")
    ex.settle(6)
    br = next((q for q in ex.s.paths if q.name != "Workflow"), None)
    print(" after denoise", state(), "branch blocks",
          [n["block"] for n in ex.s.p.workflow(br.pid).blocks] if br else None)
    shot("n4_branch_grown")


def n5():
    root = ex.project("mono")
    ex.open(root)
    ex.w.activateWindow()
    ex.double_click_card("fusion")
    ex.settle(8)
    print(" fusion placed", state())
    for fl in ("Ha", "OIII"):
        q = next(q for q in ex.s.paths if q.name == fl)
        m = ex.right_click_block(pp.block("debruitage", pid=q.pid), "Vers Post")
        print(" menu", fl, [d.get("text") for d in m])
        ex.settle(6)
        print(" toast", pp.toast())
    ex.click_block(pp.final())
    print(" layers", pp.graph_layers(), state()["paths"])
    shot("n5_channel_layers")


def n6():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("empilement"), "Vers Post")
    ex.settle(6)
    print(" toast", pp.toast(), state()["paths"])
    m = ex.right_click_block(pp.block("empilement", pid=ex.s.paths[0].pid))
    print(" menu", [d.get("text") for d in m])
    shot("n6_menu")


def n7():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    pp.pull_port(pp.block("fond"))
    print(" 1st pull", state())
    wait(6)
    pp.pull_port(pp.block("fond", pid=ex.s.paths[0].pid))
    print(" 2nd pull", state())
    m = ex.right_click_block(pp.block("fond", pid=ex.s.paths[0].pid))
    print(" menu", [d.get("text") for d in m])
    shot("n7_port_twice")


def n8():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.click_block(pp.final())
    print(" panel buttons", pp.panel_buttons())
    ex.double_click_block(pp.final())
    ex.settle(8)
    print(" mode after dbl", ex.w.mode, [t for t in ex.texts() if "calque" in t.lower()][:6])
    shot("n8_dbl_post")
    ex.w.set_mode(0)
    ex.settle(4)


def n9():
    ex.open()
    ex.w.resize(1366, 768)
    ex.settle(4)
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    vr = ex.view.mapToScene(ex.view.viewport().rect()).boundingRect()
    print(" post visible", vr.contains(pp.final().sceneBoundingRect()), "scale", ex.view.transform().m11(),
          "hbar", ex.view.horizontalScrollBar().maximum())
    shot("n9_1366")
    ex.w.resize(1600, 950)
    ex.settle(4)


def n11():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    pp.click_wire(pp.wires()[0])
    pp.del_key()
    print(" removed", pp.toast())
    wait(6)
    ex.right_click_block(pp.block("couleur"), "Vers Post")
    ex.settle(6)
    print(" sent back", state())
    shot("n11_send_back")


def n12():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    pp.click_wire(pp.wires()[0])
    pp.del_key()
    ex.click_block(pp.final())
    print(" final", pp.final_summary(), "panel", pp.panel_texts())
    print(" run buttons", [(b.text(), b.isEnabled(), b.toolTip()[:200]) for b in ex.w.findChildren(type(ex.w.mode_btns[0]))
                           if b.isVisible() and ("Exécuter" in b.text())])
    shot("n12_zero_layer")


def n13():
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    ex.click_block(pp.blocks_of("composition")[0])
    pp.del_key()
    print(" deleted", pp.toast())
    wait(6)
    m = ex.right_click_block(pp.block("etoiles"), "+ Recombiner")
    ex.settle(6)
    print(" menu", [d.get("text") for d in m])
    print(" after +Recombiner", state())
    shot("n13_recombine")


def n14():
    root = ex.project("mono")
    ex.open(root)
    ex.w.activateWindow()
    ch = lambda: {q.name: [n['block'] for n in (ex.s.p.workflow(q.pid).blocks or [])] for q in ex.s.paths}  # noqa
    print(" before", ch(), pp.graph_layers())
    ex.double_click_card("fusion")
    ex.settle(8)
    print(" fusion", ch(), pp.toast(), pp.graph_layers())
    m = ex.right_click_block(pp.block("fusion"), "Supprimer")
    ex.settle(6)
    print(" menu", [d.get("text") for d in m])
    print(" deleted", ch(), pp.toast(), pp.graph_layers(), len(pp.blocks_of("composition")))
    shot("n14_fusion_deleted")


def n15():
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    m = ex.right_click_block(pp.block("etoiles"), "Vers Post")
    ex.settle(6)
    print(" menu", [(d.get("text"), d.get("tip")) for d in m if "Post" in (d.get("text") or "")])
    print(" after", state())
    shot("n15_split_send")


def n16():
    session_project(ex)
    build_chain(ex, CHAINS["gradient"])
    pp.click_wire(pp.wires()[0])
    print(" selected", pp.wires()[0].selected)
    vp = ex.view.viewport()
    ex.click(vp, QPoint(vp.width() // 2, 40))
    s1 = pp.wires()[0].selected
    pp.click_wire(pp.wires()[0])
    pp.key(Qt.Key_Escape)
    s2 = pp.wires()[0].selected
    print(" after empty click", s1, "after esc", s2)
    pp.del_key()
    print(" after del", state())
    shot("n16_wire_select")


def n17():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    w = wire_of("Gradient")
    ex.dialog_handler = rename_to("Workflow")
    m = pp.right_click_wire(w, "Renommer")
    ex.settle(8)
    print(" wire menu", [d.get("text") for d in (m or [])])
    print(" after rename", state(), ex.popups[-2:], ex.modals[-1:])
    ex.click_block(pp.final())
    shot("n17_same_name")


def n18():
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    simulate()
    pp.pull_port(pp.block("etoiles"), "starless")
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    pp.pull_port(pp.block("empilement", pid=ex.s.paths[0].pid))
    print(" toast", pp.toast())
    v = post_page()
    print(" layers", [(lay["name"], lay.get("visible")) for lay in v.layers])
    shot("n18_visible")
    graph_page()


def n19():
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    simulate()
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    v = post_page()
    row = next(r for r in v.rows if "Gradient" in r.lay["name"])
    ex.dialog_handler = rename_to("Fond seul")
    ex.double_click(row.n)
    ex.settle(8)
    eds = [e for e in v.findChildren(QLineEdit) if e.isVisible()]
    if eds:
        eds[0].setText("Fond seul")
        from PySide6.QtTest import QTest
        QTest.keyClick(eds[0], Qt.Key_Return)
        ex.settle(6)
    print(" rows", [r.n.text() for r in v.rows])
    graph_page()
    print(" graph", state())
    shot("n19_rename_in_post")


def n20():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    v = post_page()
    lay = v.layers[0]
    v.select(lay["id"], "layer")
    ex.settle(3)
    print(" layers", [x["name"] for x in v.layers], "b_del", v.b_del.isEnabled(), repr(v.b_del.toolTip()))
    ex.hover(v.b_del, v.b_del.rect().center())
    shot("n20_delete_single")
    graph_page()


SC = {k: v for k, v in globals().items() if k[:1] == "n" and k[1:].isdigit()}
for k in sys.argv[1:] or sorted(SC, key=lambda x: int(x[1:])):
    for r in (1, 2):
        RUN[0] = r
        print(f"=== {k} run {r}", flush=True)
        try:
            SC[k]()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("FAILED", k, e)
    print("slot errors", [t.strip().splitlines()[-1] for t in pp.take_slot_errors()], flush=True)
ex.close()
