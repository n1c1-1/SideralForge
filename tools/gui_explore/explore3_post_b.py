"""post_wires round 3 (control), part B: the round-1 points still open in round 2, the Fusion points, and checks
around the fixes. Each scenario twice, from a fresh project. Prints only."""
import sys
import time

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit

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


def state():
    return {"paths": [q.name for q in ex.s.paths], "blocks": sorted(it.node.key for it in ex.blocks()),
            "wires": [w.toolTip().splitlines()[0] for w in pp.wires()], "final": pp.final_summary(),
            "layers": pp.graph_layers(), "toast": pp.toast()}


def chains():
    return {q.name: [n['block'] for n in (ex.s.p.workflow(q.pid).blocks or [])] for q in ex.s.paths}


def wire_of(part):
    return next((w for w in pp.wires() if part in w.toolTip()), None)


def post_page():
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    return ex.w.post_holder.page.view


def graph_page():
    ex.click(ex.w.mode_btns[0])
    ex.settle(8)


def selected_blocks():
    return [it.node.key for it in ex.blocks() if it.isSelected() or getattr(it, "selected", False)]


def mono_with_fusion():
    root = ex.project("mono")
    ex.open(root)
    ex.w.activateWindow()
    c = ex.card("fusion")
    print(" fusion card", c.isEnabled() if c else None, (c.toolTip() if c else "")[:300])
    for fl in ("Ha", "OIII"):
        q = next(q for q in ex.s.paths if q.name == fl)
        ex.click_block(pp.block("empilement", pid=q.pid))
        ex.double_click_card("astrometry")
        ex.settle(6)
    c = ex.card("fusion")
    print(" chains with astrometry", chains(), "fusion card", c.isEnabled() if c else None,
          (c.toolTip() if c else "")[:300])
    before = (chains(), pp.graph_layers())
    ex.double_click_card("fusion")
    ex.settle(8)
    print(" fusion placed", chains(), "toast", pp.toast(), "layers", pp.graph_layers())
    return before


def m_fusion_delete():                       # round-2 new #12
    before = mono_with_fusion()
    f = pp.block("fusion")
    if f is None:
        print(" NO FUSION BLOCK", state())
        return
    m = ex.right_click_block(f, "Supprimer")
    ex.settle(6)
    print(" menu", [d.get("text") for d in m])
    print(" deleted", chains(), "toast", pp.toast(), "layers", pp.graph_layers(), "comp", len(pp.blocks_of("composition")))
    print(" same as before fusion:", (chains(), pp.graph_layers()) == before)
    shot("m_fusion_deleted")
    pp.undo()
    print(" undo", chains(), pp.graph_layers(), "fusion", pp.block("fusion") is not None)


def m_channels_send():                       # round-2 new #4 and round-1 #36
    mono_with_fusion()
    for fl in ("Ha", "OIII"):
        q = next(q for q in ex.s.paths if q.name == fl)
        last = [n['block'] for n in ex.s.p.workflow(q.pid).blocks if n['block'] != "output"][-1]
        key = {"denoise": "debruitage", "astrometry": "astrometrie", "stars": "etoiles", "stacking": "empilement"}.get(last, last)
        it = pp.block(key, pid=q.pid)
        m = ex.right_click_block(it, "Vers Post")
        print(" menu", fl, key, [d.get("text") for d in m])
        ex.settle(6)
        print(" toast", pp.toast())
    print(" layers", pp.graph_layers(), "paths", [q.name for q in ex.s.paths])
    shot("m_channels_send")


def r_post_menu():                           # R1, R2, R3
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    m = ex.right_click_block(pp.final())
    print(" post menu", [d.get("text") for d in m])
    ex.click_block(pp.block("empilement"))
    pp.click_wire(pp.wires()[0])
    print(" after wire click selected blocks", selected_blocks(), "wire sel", pp.wires()[0].selected,
          "panel", pp.panel_texts()[:3])
    shot("r_wire_selected")
    ex.double_click_block(pp.final())
    ex.settle(8)
    v = ex.w.post_holder.page.view
    print(" post page texts", [t for t in ex.texts(ex.w.post_holder) if t][:12])
    shot("r_post_page")
    graph_page()


def r_combine():                             # R5, R20, R21, R38, R39
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    m = ex.right_click_block(pp.blocks_of("composition")[0])
    print(" combine menu", [d.get("text") for d in m])
    ex.click_block(pp.blocks_of("composition")[0])
    print(" combine panel buttons", pp.panel_buttons())
    b = [x for x in ex.page.panel.findChildren(type(ex.w.mode_btns[0])) if x.isVisible() and "Post" in x.text()]
    if b:
        print(" open-in-post button", b[0].text(), b[0].isEnabled(), b[0].toolTip())
        if b[0].isEnabled():
            ex.click(b[0])
            ex.settle(10)
            print(" mode", ex.w.mode, [t for t in ex.texts(ex.w.post_holder) if t][:8])
            shot("r_combine_open")
            graph_page()
    # a second Combine
    m = ex.right_click_block(pp.block("etoiles"), "+ Recombiner")
    ex.settle(6)
    print(" etoiles menu", [d.get("text") for d in m])
    print(" after 2nd recombine", [(it.title, it.summary) for it in pp.blocks_of("composition")], pp.graph_layers(),
          pp.toast())
    ex.click_block(pp.final())
    print(" post panel", pp.panel_texts()[:6])
    shot("r_two_combines")
    # split panel buttons
    ex.click_block(pp.block("etoiles"))
    print(" split panel buttons", pp.panel_buttons())
    for txt in ("Hors Post", "Vers Post"):
        bs = [x for x in ex.page.panel.findChildren(type(ex.w.mode_btns[0])) if x.isVisible() and x.text() == txt]
        if bs:
            wait(6)
            ex.click(bs[0])
            ex.settle(6)
            print(f" split panel '{txt}' ->", state()["wires"], "toast", pp.toast())
            shot(f"r_split_{txt.replace(' ', '_')}")
            ex.click_block(pp.block("etoiles"))


def r_channels_undo():                       # R16
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    simulate()
    v = post_page()
    lay = next(lay for lay in v.layers if lay["kind"] == "workflow")
    v.select(lay["id"], "layer")
    ex.settle(4)
    cs = [c for c in v.right.findChildren(QComboBox) if c.isVisible()]
    print(" combos", [[c.itemText(i) for i in range(c.count())] for c in cs])
    ch = next((c for c in cs if c.count() == 2), None)
    if ch is None:
        print(" no 2-item combo; checkboxes:", [(x.text(), x.isChecked()) for x in v.right.findChildren(
            type(ex.w.mode_btns[0])) if x.isVisible() and x.isCheckable()])
        return
    before = ch.currentText()
    ch.setFocus()
    ex.settle(1)
    QTest.keyClick(ch, Qt.Key_Down)
    ex.settle(8)
    mid = ch.currentText()
    pp.undo()
    ex.settle(8)
    c2 = [x for x in v.right.findChildren(QComboBox) if x.isVisible() and x.count() == 2]
    print(" channels", before, "->", mid, "-> after Ctrl+Z", [x.currentText() for x in c2])
    shot("r_channels_undo")
    graph_page()


def r_shared_remove():                       # R27
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    m = ex.right_click_block(pp.block("fond"), "Supprimer dans 2 workflows")
    ex.settle(6)
    print(" menu", [d.get("text") for d in m])
    print(" after", state(), chains())
    shot("r_shared_remove")


def r_image_chain():                         # R34
    root = ex.project("image")
    ex.open(root)
    ex.w.activateWindow()
    ex.click_block(pp.block("image"))
    for b in ("astrometry", "gradient", "color"):
        ex.double_click_card(b)
        ex.settle(6)
    im = pp.block("image")
    m = ex.right_click_block(im)
    print(" image menu", [d.get("text") for d in m], "ports", im.ports, chains())
    if any(d.get("text") == "Vers Post" for d in m):
        ex.right_click_block(pp.block("image"), "Vers Post")
        ex.settle(6)
        print(" sent", state())
    shot("r_image_chain")


def r_split_menu_send():                     # round-2 new #13: what the Split menu items send
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    print(" start", state())
    for it in ("Sans étoiles vers Post", "Étoiles vers Post"):
        ex.right_click_block(pp.block("etoiles"), it)
        ex.settle(6)
        print(f" after '{it}'", state())
    lays = (getattr(ex.s.p, "post", None) or {}).get("layers") or []
    print(" saved", [(x.get("kind"), x.get("part"), x.get("name")) for x in lays])
    shot("r_split_menu_send")


def r_shared_pull_name():                    # check: shared Empilement pulled after a branch exists
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    pp.pull_port(pp.block("empilement"))
    print(" after pull", state())
    ex.right_click_block(pp.block("empilement"), "Vers Post")
    ex.settle(6)
    print(" after menu", state())
    shot("r_shared_pull_name")


def r_more():                               # Split's workflow submenu, Post panel names, channel Denoise, Canaux
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    m = ex.right_click_block(pp.block("etoiles"))
    print(" etoiles menu full", m[-1])
    ex.right_click_block(pp.block("etoiles"), "Sans étoiles vers Post")
    ex.settle(6)
    ex.right_click_block(pp.block("etoiles"), "Étoiles vers Post")
    ex.settle(6)
    ex.click_block(pp.final())
    print(" post panel", pp.panel_texts())
    shot("r_post_panel_names")
    pp.click_wire(pp.wires()[0])
    print(" wire selected panel", pp.panel_texts(), [w.selected for w in pp.wires()])
    shot("r_wire_panel")
    simulate()
    v = post_page()
    lay = next(lay for lay in v.layers if lay["kind"] == "workflow")
    v.select(lay["id"], "layer")
    ex.settle(4)
    from PySide6.QtWidgets import QAbstractButton, QLabel
    print(" post right", [x.text() for x in v.right.findChildren(QLabel) if x.isVisible() and x.text()],
          [(x.text(), x.isChecked()) for x in v.right.findChildren(QAbstractButton) if x.isVisible()])
    shot("r_post_right")
    graph_page()


def m_channel_denoise():
    mono_with_fusion()
    q = next(q for q in ex.s.paths if q.name == "Ha")
    m = ex.right_click_block(pp.block("debruitage", pid=q.pid), "Vers Post")
    ex.settle(6)
    print(" menu", [d.get("text") for d in m], "toast", pp.toast())
    q = next(q for q in ex.s.paths if q.name == "OIII")
    ex.right_click_block(pp.block("debruitage", pid=q.pid), "Vers Post")
    ex.settle(6)
    print(" toast", pp.toast(), "layers", pp.graph_layers(), "paths", [q.name for q in ex.s.paths])
    ex.click_block(pp.final())
    print(" post panel", pp.panel_texts())
    shot("m_channel_denoise")


SC = {k: v for k, v in globals().items() if k[:2] in ("r_", "m_")}
for k in sys.argv[1:] or list(SC):
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
