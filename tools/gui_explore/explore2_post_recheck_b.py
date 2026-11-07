"""post_wires round 2, recheck B: the Output block of a branch (#7 #9 #10 #12), Combine's Open in Post (#6), the Post
page's own undo of Channels (#17, simulated computed state)."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtWidgets import QAbstractButton, QComboBox, QLabel, QWidget, QCheckBox

start(900)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["output", "combine", "page"]


def R(n, *a):
    print(f"R{n}:", *a, flush=True)


def paths():
    return [(q.name, [n['block'] for n in ex.s.p.workflow(q.pid).blocks]) for q in ex.s.paths]


def vis(root):
    out = []
    for x in [root] + root.findChildren(QWidget):
        if not x.isVisible():
            continue
        if isinstance(x, (QLabel, QAbstractButton)) and x.text():
            out.append(x.text())
        if isinstance(x, QComboBox):
            out.append("COMBO:" + "/".join(x.itemText(i) for i in range(x.count())))
    return out


def post_page():
    return ex.w.post_holder.page if ex.w.post_holder is not None else None


if "output" in which:
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    out = pp.blocks_of("post")
    R(5, "output blocks", [(o.title, o.summary, o.toolTip()) for o in out])
    o = out[0]
    m = ex.right_click_block(o)
    R(7, "output menu", [d.get("text") for d in m])
    ex.shot("recheck_07_output_menu") if False else None
    o = pp.blocks_of("post")[0]
    ex.click_block(o)
    R(10, "output panel", pp.panel_texts())
    ex.shot("recheck_10_output_panel")
    pp.del_key()
    R(9, "Del on output: toast", pp.toast(), "paths", paths(), "post", pp.final_summary(), "popups", ex.popups[-2:])
    ex.shot("recheck_09_del_output")
    pp.undo()
    R(9, "undo", paths(), pp.final_summary())
    o = pp.blocks_of("post")[0] if pp.blocks_of("post") else None
    if o is not None:
        n = len(ex.popups)
        nm = len(ex.modals)
        ex.right_click_block(o, "Supprimer workflow")
        ex.settle(6)
        R(12, "Supprimer workflow on output: popups", ex.popups[n:], "modals", ex.modals[nm:], "toast", pp.toast(),
          "paths", paths())
        pp.undo()
        R(12, "undo", paths(), pp.final_summary())
    # on the main workflow's last block (Couleur): Supprimer workflow?
    m = ex.right_click_block(pp.block("couleur"))
    R(12, "couleur menu", [d.get("text") for d in m])

if "combine" in which:
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    cb = pp.blocks_of("composition")[0]
    m = ex.right_click_block(cb)
    R(6, "combine menu", m)
    R(22, "combine menu texts", [d.get("text") for d in m])
    pick = next((d["text"] for d in m if "Post" in (d.get("text") or "") or "Ouvrir" in (d.get("text") or "")), None)
    if pick:
        ex.right_click_block(pp.blocks_of("composition")[0], pick)
        ex.settle(6)
        pg = post_page()
        R(6, "after", pick, "mode", ex.w.mode, "page idx", pg.pages.currentIndex() if pg else None,
          vis(pg)[:6] if pg else None)
        ex.shot("recheck_06_combine_open")
        ex.w.set_mode(0)
        ex.settle(4)

if "page" in which:
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    s = ex.s
    orig_sync = s._sync

    def fake_sync(*a, **k):
        orig_sync(*a, **k)
        for v in s.paths:
            v.rendered = True
            v.computed = False
    s._sync = fake_sync
    s._sync()
    pp.pull_port(pp.block("etoiles"), "starless")
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)
    pp.dump("simulated")
    ex.click_block(pp.final())
    R(1, "post panel sim", pp.panel_texts(), pp.panel_buttons())
    ex.click(ex.w.mode_btns[2])
    ex.settle(20)
    pg = post_page()
    v = pg.view
    R(3, "page idx", pg.pages.currentIndex(), "rows", [r.n.text() for r in v.rows],
      "layers", [(lay["kind"], lay.get("part"), lay["name"]) for lay in v.layers])
    ex.shot("recheck_post_page_sim")
    lay = next(lay for lay in v.layers if lay["kind"] == "workflow")
    v.select(lay["id"], "layer")
    ex.settle(4)
    R(10, "layer props", vis(v.right)[:30])
    combos = [c for c in v.right.findChildren(QComboBox) if c.isVisible() and c.count() == 2]
    checks = [c for c in v.right.findChildren(QCheckBox) if c.isVisible()]
    R(10, "combos", [[c.itemText(i) for i in range(c.count())] for c in combos], "checks", [c.text() for c in checks])
    ex.shot("recheck_10_layer_props")
    if combos:
        c = combos[0]
        before = c.currentIndex()
        c.setCurrentIndex(1 - before)
        ex.settle(6)
        pp.undo()
        ex.settle(6)
        combos2 = [x for x in v.right.findChildren(QComboBox) if x.isVisible() and x.count() == 2]
        R(17, "channels before", before, "after Ctrl+Z", [x.currentIndex() for x in combos2],
          [x.currentText() for x in combos2])
        pp.redo()
        ex.settle(6)
        combos2 = [x for x in v.right.findChildren(QComboBox) if x.isVisible() and x.count() == 2]
        R(17, "after Ctrl+Y", [x.currentIndex() for x in combos2])
    elif checks:
        c = checks[0]
        before = c.isChecked()
        ex.click(c)
        ex.settle(6)
        pp.undo()
        ex.settle(6)
        checks2 = [x for x in v.right.findChildren(QCheckBox) if x.isVisible()]
        R(17, "check before", before, "after Ctrl+Z", [x.isChecked() for x in checks2])
        pp.redo()
        ex.settle(6)
        checks2 = [x for x in v.right.findChildren(QCheckBox) if x.isVisible()]
        R(17, "after Ctrl+Y", [x.isChecked() for x in checks2])
    ex.w.set_mode(0)
    ex.settle(6)
    pp.dump("back to graph")
print("popups", ex.popups, "modals", ex.modals, "slot", pp.take_slot_errors())
ex.close()


