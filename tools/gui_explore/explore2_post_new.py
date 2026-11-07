"""post_wires round 2: new explorations (two Image blocks without file, a send branch grown then its wire removed,
renaming a branch, the Output's Channels vs the Post layer, Image alone wire removed, Fusion sent, duplicate)."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QCheckBox, QComboBox, QLineEdit

start(900)
ex = Explorer("post_wires", lang=(sys.argv[1] if len(sys.argv) > 1 else "fr"))
pp = P(ex)
which = sys.argv[2:] or ["images", "grown", "rename", "channels", "image_alone", "fusion", "dup"]


def R(n, *a):
    print(f"{n}:", *a, flush=True)


def paths():
    return [(q.name, [n['block'] for n in (ex.s.p.workflow(q.pid).blocks or [])]) for q in ex.s.paths]


def st(tag=""):
    return (f"[{tag}] post={pp.final_summary()!r} layers={pp.graph_layers()} "
            f"wires={[w.toolTip().splitlines()[0] for w in pp.wires()]} toast={pp.toast()!r}")


def rename_to(name):
    def h(d):
        e = d.findChildren(QLineEdit)
        if e:
            e[0].setText(name)
            d.accept()
        else:
            d.reject()
    return h


if "images" in which:
    ex.new_project("DeuxImages")
    ex.w.activateWindow()
    ex.double_click_card("image")
    ex.settle(8)
    R("images", st("1 image"), paths(), pp.blocks_desc())
    ex.double_click_card("image")
    ex.settle(8)
    R("images", st("2nd image dbl"), paths(), pp.blocks_desc())
    ex.shot("new_two_images_nofile")
    pp.undo()
    R("images", st("undo"), paths(), pp.blocks_desc())
    pp.redo()
    R("images", st("redo"), paths(), pp.blocks_desc())

if "grown" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    br = ex.s.paths[-1]
    # select the branch (its Output block) and add Débruitage to it by palette double-click
    ex.click_block(pp.blocks_of("post")[0])
    R("grown", "palette for", [c for c in ex.page.cards if ex.page.cards[c].isEnabled()])
    ex.double_click_card("denoise")
    ex.settle(6)
    R("grown", st("denoise added to branch"), paths(), pp.blocks_desc())
    ex.shot("new_branch_grown")
    w = next((w for w in pp.wires() if br.name in w.toolTip()), None)
    if w is not None and pp.click_wire(w):
        pp.del_key()
        R("grown", st("its wire removed"), paths())
        ex.shot("new_branch_grown_wire_removed")
        pp.undo()
        R("grown", st("undo"), paths())

if "rename" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    ex.dialog_handler = rename_to("Fond propre")
    m = ex.right_click_block(pp.blocks_of("post")[0], "Renommer")
    ex.settle(8)
    R("rename", st("branch renamed"), paths(), [(b.title, b.summary) for b in pp.blocks_of("post")])
    ex.click_block(pp.final())
    R("rename", "post panel", pp.panel_texts()[2:3])
    ex.shot("new_branch_renamed")
    pp.undo()
    R("rename", st("undo"), paths())
    # rename the main workflow (via Couleur's menu)
    ex.dialog_handler = rename_to("Principal")
    ex.right_click_block(pp.block("couleur"), "Renommer")
    ex.settle(8)
    R("rename", st("main renamed"), paths())

if "channels" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    s = ex.s
    orig = s._sync

    def fake(*a, **k):
        orig(*a, **k)
        for v in s.paths:
            v.rendered = True
            v.computed = False
    s._sync = fake
    s._sync()
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    ex.click_block(pp.blocks_of("post")[0])
    cb = [c for c in ex.page.panel.findChildren(QCheckBox) if c.isVisible()]
    R("channels", "output panel checks", [(c.text(), c.isChecked()) for c in cb])
    if cb:
        ex.click(cb[0])
        ex.settle(6)
        R("channels", "after click", [(c.text(), c.isChecked()) for c in cb])
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    v = ex.w.post_holder.page.view
    lay = next(lay for lay in v.layers if "Gradient" in lay["name"])
    v.select(lay["id"], "layer")
    ex.settle(4)
    cs = [c for c in v.right.findChildren(QComboBox) if c.isVisible() and c.count() == 2]
    R("channels", "post layer channels", [c.currentText() for c in cs], {k: lay[k] for k in lay if k not in ("filters",)})
    ex.shot("new_channels_post")
    if cs:
        cs[0].setCurrentIndex(1 - cs[0].currentIndex())
        ex.settle(6)
    ex.click(ex.w.mode_btns[0])
    ex.settle(8)
    ex.click_block(pp.blocks_of("post")[0])
    cb = [c for c in ex.page.panel.findChildren(QCheckBox) if c.isVisible()]
    R("channels", "output panel after Post change", [(c.text(), c.isChecked()) for c in cb])
    # the main workflow's channels: where?
    ex.click_block(pp.block("couleur"))
    R("channels", "couleur panel checks", [(c.text(), c.isChecked()) for c in ex.page.panel.findChildren(QCheckBox)
                                            if c.isVisible()])
    ex.click_block(pp.final())
    R("channels", "post block panel", pp.panel_texts())

if "image_alone" in which:
    root = ex.project("image")
    ex.open(root)
    ex.w.activateWindow()
    R("image_alone", st("image alone"))
    w = pp.wires()[0] if pp.wires() else None
    if w is not None and pp.click_wire(w):
        pp.del_key()
        R("image_alone", st("wire removed"), paths())
        m = ex.right_click_block(pp.block("image"))
        R("image_alone", "image menu", [d.get("text") for d in m])
        ex.click_block(pp.block("image"))
        R("image_alone", "panel", pp.panel_buttons())
        ex.shot("new_image_alone_wire_removed")
        pp.undo()
        R("image_alone", st("undo"))
    for b in ("astrometry", "gradient"):
        ex.double_click_card(b)
        ex.settle(6)
    R("image_alone", st("astrometry gradient"), paths())
    m = ex.right_click_block(pp.block("astrometrie"))
    R("image_alone", "astrometry menu", [d.get("text") for d in m])

if "fusion" in which:
    root = ex.project("mono")
    ex.open(root)
    ex.w.activateWindow()
    ex.double_click_card("fusion")
    ex.settle(8)
    R("fusion", st("fusion"), paths())
    fu = pp.block("fusion")
    m = ex.right_click_block(fu)
    R("fusion", "fusion menu", [d.get("text") for d in m], "ports", fu.ports)
    for key in ("etoiles",):
        for it in pp.blocks_of(key):
            m = ex.right_click_block(it)
            R("fusion", "channel split menu", it.summary, [d.get("text") for d in m])
            break
    q = next(q for q in ex.s.paths if q.name == "OIII")
    ex.right_click_block(pp.block("debruitage", pid=q.pid), "Vers Post")
    ex.settle(6)
    R("fusion", st("OIII denoise sent"), paths())
    ex.click_block(pp.final())
    R("fusion", "post panel", pp.panel_texts()[2:3])
    ex.shot("new_fusion_channel_sent")
    pp.pull_port(pp.block("fusion"))
    R("fusion", st("fusion port pulled"), paths())
    comp = pp.blocks_of("composition")
    if comp:
        ex.click_block(comp[0])
        pp.del_key()
        R("fusion", st("fusion's Recombiner deleted"), paths())
        m = ex.right_click_block(pp.block("fusion"))
        R("fusion", "fusion menu now", [d.get("text") for d in m])
        ex.shot("new_fusion_recombine_deleted")
        pp.undo()
        R("fusion", st("undo"))

if "dup" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    ex.right_click_block(pp.block("couleur"), "Dupliquer")
    ex.settle(8)
    R("dup", st("main duplicated"), paths())
    ex.shot("new_dup")
    pp.undo()
    R("dup", st("undo"), paths())
    ex.right_click_block(pp.blocks_of("post")[0], "Dupliquer")
    ex.settle(8)
    R("dup", st("branch duplicated"), paths(), [(b.title, b.summary) for b in pp.blocks_of("post")])
print("popups", ex.popups, "modals", ex.modals, "slot", pp.take_slot_errors())
ex.close()
