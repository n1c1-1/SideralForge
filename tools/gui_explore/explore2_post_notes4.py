"""post_wires round 2, fourth batch: each candidate reproduced twice before it is noted."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QLineEdit

start(900)
ex = Explorer("post_wires")
pp = P(ex)
which = sys.argv[1:] or ["n16", "n17"]


def n16():
    session_project(ex)
    build_chain(ex, CHAINS["gradient"])
    pp.click_wire(pp.wires()[0])
    vp = ex.view.viewport()
    ex.click(vp, QPoint(vp.width() // 2, 40))
    still = pp.wires()[0].selected
    pp.key(Qt.Key_Escape)
    still2 = pp.wires()[0].selected
    f = ex.shot("n16_wire_still_selected")
    pp.del_key()
    gone = not pp.wires()
    return (still and still2 and gone), ("Clic sur le fil vers Post (il s'allume), puis clic dans le vide du graphe, "
                                         "puis Échap : le fil reste sélectionné ; Suppr le retire alors (« « Workflow » "
                                         "hors Post »). Dans les éditeurs de référence, un clic dans le vide ou Échap "
                                         "désélectionne."), f


def n17():
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    ex.right_click_block(pp.block("fond"), "Vers Post")
    ex.settle(6)

    def h(d):
        e = d.findChildren(QLineEdit)
        e[0].setText("Workflow")
        d.accept()
    ex.dialog_handler = h
    ex.right_click_block(pp.blocks_of("post")[0], "Renommer")
    ex.settle(8)
    names = [q.name for q in ex.s.paths]
    ex.click_block(pp.final())
    f = ex.shot("n17_two_layers_same_name")
    return (names.count("Workflow") == 2), ("Gradient envoyé au Post (branche « Sortie Gradient »), puis la branche "
                                            "renommée « Workflow », le nom du workflow principal : accepté sans "
                                            "message ; le panneau du bloc Post liste deux calques « Workflow », qu'on "
                                            "ne peut plus distinguer."), f


CANDS = {
    "n16": ("mineur", "Fil sélectionné : un clic dans le vide ou Échap ne le désélectionne pas",
            ["Nouveau projet avec une session de 15 brutes", "Double-clic dans la palette : Calibration … Gradient",
             "Clic sur le fil vers Post", "Clic dans le vide du graphe", "Échap", "Suppr"], n16),
    "n17": ("mineur", "Deux workflows peuvent porter le même nom : deux calques identiques dans Post",
            ["Nouveau projet avec une session de 15 brutes", "Double-clic dans la palette : Calibration … Couleur",
             "Clic droit sur Gradient > Vers Post", "Clic droit sur le bloc Sortie > Renommer : Workflow",
             "Clic sur Post"], n17),
}
for k in which:
    sev, title, steps, fn = CANDS[k]
    a = fn()
    b = fn()
    print(k, a[0], b[0], "|", b[1], flush=True)
    if a[0] and b[0]:
        ex.note(sev, title, b[1], shot=b[2], steps=steps)
    else:
        print("NOT REPRODUCED", k)
ex.close()
