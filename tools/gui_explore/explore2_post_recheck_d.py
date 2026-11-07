"""post_wires round 2, recheck D: rename (#38), sources (#33 #34 #35), two flows (#36), Fusion channel (#37), the
shared block deleted after a send (#27 #28)."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtWidgets import QLineEdit

start(900)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["rename", "image", "empty", "flows", "mono", "shared"]


def R(n, *a):
    print(f"R{n}:", *a, flush=True)


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


if "rename" in which:
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    pp.pull_port(pp.block("etoiles"), "starless")
    ex.dialog_handler = rename_to("M31 final")
    ex.right_click_block(pp.block("etoiles"), "Renommer")
    ex.settle(8)
    R(38, "paths", [q.name for q in ex.s.paths])
    ex.click_block(pp.final())
    R(38, st("renamed"), pp.panel_texts()[2])
    ex.shot("recheck_38_renamed")
    pp.undo()
    R(38, st("undo"), [q.name for q in ex.s.paths])
    pp.redo()
    R(38, st("redo"), [q.name for q in ex.s.paths])

if "image" in which:
    ex.new_project("Vide")
    ex.w.activateWindow()
    ex.double_click_card("image")
    ex.settle(8)
    R(33, st("image no file"), paths())
    ex.shot("recheck_33_image_nofile")
    root = ex.project("image")
    ex.open(root)
    ex.w.activateWindow()
    R(35, st("image source alone"), paths())
    for b in ("astrometry", "gradient", "color"):
        ex.double_click_card(b)
        ex.settle(6)
    im = pp.block("image")
    m = ex.right_click_block(im)
    R(35, "image menu", [d.get("text") for d in m], "ports", pp.block("image").ports)
    if any(d.get("text") == "Vers Post" for d in m):
        ex.right_click_block(pp.block("image"), "Vers Post")
        ex.settle(6)
        R(35, st("image sent"), paths())
        ex.shot("recheck_35_image_sent")
        pp.undo()
        R(35, st("undo"), paths())

if "empty" in which:
    ex.new_project("VideSession")
    ex.w.activateWindow()
    ex.double_click_card("session")
    ex.settle(8)
    build_chain(ex, ["calibration", "registration", "stacking"])
    R(34, st("empty session chain"), paths())
    ex.shot("recheck_34_empty_session")

if "flows" in which:
    root = ex.project("two_flows")
    ex.open(root)
    ex.w.activateWindow()
    for pid in [q.pid for q in ex.s.paths]:
        for b in CHAINS["gradient"]:
            ex.page.place_block(b, pid)
            ex.settle(4)
    R(36, st("two flows"))
    w2 = [w for w in pp.wires() if "Flow" in w.toolTip()]
    if w2:
        pp.click_wire(w2[0])
        pp.del_key()
        R(36, st("flow 2 wire removed"))
        p1 = ex.s.paths[0]
        pp.pull_port(pp.block("empilement", pid=p1.pid))
        R(36, st("flow1 stacking sent"), paths())
        ex.shot("recheck_36_two_flows")

if "mono" in which:
    root = ex.project("mono")
    ex.open(root)
    ex.w.activateWindow()
    ex.double_click_card("fusion")
    ex.settle(8)
    R(37, st("fusion placed"), paths())
    q = next(q for q in ex.s.paths if q.name == "Ha")
    m = ex.right_click_block(pp.block("empilement", pid=q.pid))
    R(37, "Ha stacking menu", [d.get("text") for d in m], "ports", pp.block("empilement", pid=q.pid).ports)
    if any(d.get("text") == "Vers Post" for d in m):
        ex.right_click_block(pp.block("empilement", pid=q.pid), "Vers Post")
        ex.settle(6)
        R(37, st("Ha sent"), paths())
        ex.shot("recheck_37_channel_sent")

if "shared" in which:
    for mid in ("stacking", "gradient"):
        key = {"stacking": "empilement", "gradient": "fond"}[mid]
        session_project(ex)
        build_chain(ex, CHAINS["color"])
        pp.pull_port(pp.block(key))
        R(27, mid, st("sent by port"), paths())
        m = ex.right_click_block(pp.block(key, pid=ex.s.paths[0].pid))
        texts = [d.get("text") for d in m]
        R(27, "menu shared", texts)
        pick = next((t for t in texts if t and ("Supprimer" in t or "Retirer" in t)), None)
        if pick:
            ex.right_click_block(pp.block(key, pid=ex.s.paths[0].pid), pick.split("\t")[0])
            ex.settle(6)
            R(27, st("after " + pick), paths())
            ex.shot(f"recheck_27_shared_{mid}")
            pp.undo()
            R(27, st("undo"), paths())
print("popups", ex.popups, "modals", ex.modals, "slot", pp.take_slot_errors())
ex.close()

