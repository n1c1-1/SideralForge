"""post_wires H: a workflow renamed after its images went to Post (layer names follow?); the Split panel's own buttons
(To Post / Out of Post); a Combine block's panel Delete then Ctrl+Z; at 1366x768 the Post block's panel."""
from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start, steps_chain
from PySide6.QtWidgets import QLineEdit, QPushButton

start(1200)
ex = Explorer("post_wires", size=(1366, 768))
pp = P(ex)
ST = steps_chain("stars")


def rename_to(name):
    def h(d):
        e = d.findChildren(QLineEdit)
        if e:
            e[0].setText(name)
        d.accept()
    return h


session_project(ex)
build_chain(ex, CHAINS["stars"])
ex.right_click_block(pp.block("etoiles"), "Sans étoiles vers Post")
print("before rename:", pp.graph_layers())
ex.dialog_handler = rename_to("M31 final")
ex.right_click_block(pp.block("etoiles"), "Renommer")
ex.settle(6)
pp.check_slot_errors("Renommer le workflow", ST)
print("paths:", [q.name for q in ex.s.paths], "modals:", ex.modals[-1:])
gl = pp.graph_layers()
print("graph layers:", gl, "wires:", [w.toolTip().splitlines()[0] for w in pp.wires()])
ex.click_block(pp.final())
pt = pp.panel_texts()
print("post panel:", pt)
if any("Workflow ·" in t for t in pt) and ex.s.paths[0].name == "M31 final":
    pp.note_once("rename_layer_stale", "gênant", "Workflow renommé : ses calques du Post-traitement gardent l'ancien nom",
                 "Après « Renommer » (M31 final), le panneau du bloc Post-traitement liste encore « Workflow · Image sans "
                 "étoiles » et « Workflow · Combiner les calques étoiles et fond ». Le fil, lui, dit le nouveau nom.",
                 ST + ["Clic droit sur Split > Envoyer l'image sans étoiles à Post-traitement", "Clic droit sur Split > Renommer : M31 final",
                       "Clic sur Post-traitement"])
ex.shot("rename_1366")
# Split panel buttons
ex.click_block(pp.block("etoiles"))
btns = [b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible()]
print("split panel buttons:", [(b.text(), b.toolTip()) for b in btns])
out = [b for b in btns if b.text() == "Hors Post"]
t0 = pp.toast()
if out:
    n0 = len(pp.graph_layers())
    ex.click(out[0])
    ex.settle(6)
    pp.check_slot_errors("panneau Split > Retirer du Post", ST)
    print("after panel out:", pp.graph_layers(), "toast", pp.toast())
    if len(pp.graph_layers()) != n0 - 1:
        pp.note_once("panel_out", "gênant", "Panneau de Split : « Retirer du Post » ne retire pas l'image", "", ST)
    if pp.toast() == t0:
        pp.note_once("panel_out_silent", "mineur", "Panneau de Split : « Retirer du Post » ne dit rien et n'offre pas "
                     "Ctrl+Z", "Le menu et le fil font un message (« … retirée du Post ») ; le bouton du panneau, aucun.",
                     ST + ["Clic droit sur Split > Envoyer l'image sans étoiles", "Clic sur Split",
                           "Panneau : « Retirer du Post »"])
btns = [b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible()]
to = [b for b in btns if b.text() == "Vers Post"]
t0 = pp.toast()
if to:
    ex.click(to[0])
    ex.settle(6)
    print("after panel to:", pp.graph_layers(), "toast", pp.toast())
    if pp.toast() == t0:
        pp.note_once("panel_to_silent", "mineur", "Panneau de Split : « Vers le Post » ne dit rien",
                     "Tirer le port ou le menu affichent « Nouveau calque du Post-traitement : … » ; le bouton du panneau "
                     "n'affiche aucun message.", ST + ["Clic sur Split", "Panneau : « Vers le Post »"])
# Combine panel: Delete block, Ctrl+Z
cb = pp.blocks_of("composition")[0]
ex.click_block(cb)
db = [b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible() and b.text() == "Supprimer"]
n0 = len(pp.blocks_of("composition"))
if db:
    ex.click(db[0])
    ex.settle(6)
    print("combine deleted by panel:", len(pp.blocks_of("composition")), "toast", pp.toast())
    if len(pp.blocks_of("composition")) == n0 - 1 and "Ctrl+Z" not in pp.toast():
        pp.note_once("combine_del_no_toast", "mineur", "Bloc Combiner supprimé sans message ni Ctrl+Z proposé",
                     f"Message affiché : « {pp.toast()} ». Les autres suppressions disent « … · Ctrl+Z annule ».",
                     ST + ["Clic sur Combiner", "Panneau : Supprimer le bloc"])
ex.shot("post_1366")
ex.close()
