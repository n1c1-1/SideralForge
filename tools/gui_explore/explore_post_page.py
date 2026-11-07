"""post_wires D: the Post block's panel and the Post page with nothing computed (how it is presented), going back to
the graph; then (a simulated state: the workflows marked computed, no pixels) the layer panel: names, the Channels
setting of a workflow layer, a layer deleted in Post then the graph (a wire removed stays removed)."""
from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start, steps_chain
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QComboBox, QLabel, QWidget

start(2400)
ex = Explorer("post_wires")
pp = P(ex)


def visible_texts(root):
    out = []
    for x in [root] + root.findChildren(QWidget):
        if not x.isVisible():
            continue
        if isinstance(x, (QLabel, QAbstractButton)) and x.text():
            out.append(x.text())
        if x.toolTip():
            out.append(x.toolTip())
        if isinstance(x, QComboBox):
            out += [x.itemText(i) for i in range(x.count())]
    return out


def post_page():
    return ex.w.post_holder.page if ex.w.post_holder is not None else None


# ---------------------------------------------------------------- 1. nothing computed
for end in ("color", "stars"):
    print("=" * 20, end)
    session_project(ex)
    build_chain(ex, CHAINS[end])
    st = steps_chain(end)
    if end == "stars":
        pp.pull_port(pp.block("etoiles"), "starless")
    ex.click_block(pp.final())
    btn = [b for b in pp.panel_buttons() if "Post" in b[0]]
    print("post panel:", pp.panel_texts(), btn)
    if btn and not btn[0][1] and not btn[0][2]:
        pp.note_once("open_post_grey", "mineur", "« Ouvrir le Post-traitement » grisé sans dire pourquoi",
                     "Rien n'est calculé : le bouton du panneau du bloc Post-traitement est grisé, sans infobulle ni "
                     "message (la règle : un élément grisé dit pourquoi).", st + ["Clic sur le bloc Post-traitement"])
    summ = pp.final_summary()
    mf = ex.right_click_block(pp.final())
    print("Post block menu:", mf)
    if not mf:
        pp.note_once("post_block_no_menu", "question", "Le bloc Post-traitement n'a pas de menu contextuel",
                     "Clic droit sur le bloc Post-traitement : aucun menu (pas d'« Ouvrir le Post-traitement », rien sur "
                     "ses calques), alors que tous les autres blocs en ont un.", st + ["Clic droit sur Post-traitement"])
    # double-click on the Post block
    ex.double_click_block(pp.final())
    print("after double-click Post: mode", ex.w.mode)
    pg = post_page()
    if pg is not None:
        idx = pg.pages.currentIndex()
        tx = visible_texts(pg)
        print("post page index", idx, "texts:", tx[:20])
        pp.check_texts("page Post-traitement", tx, st)
        if idx == 0 and summ and not summ.startswith("0"):
            pp.note_once("post_empty_vs_count", "gênant",
                         "Le bloc Post-traitement annonce des calques, la page Post-traitement dit « Pas encore de calque »",
                         f"Le bloc dit « {summ} » et son panneau liste les calques (« Pas encore calculé »), mais la page "
                         "Post-traitement (double-clic sur le bloc, ou le bouton Post-traitement en haut) affiche « Aucun "
                         "calque pour l'instant » sans les nommer ni dire qu'il faut les calculer.",
                         st + ["Double-clic sur le bloc Post-traitement"])
        ex.shot(f"post_page_empty_{end}")
        # the empty page's link back
        back = [b for b in pg.findChildren(QAbstractButton) if b.isVisible() and "graph" in b.text().lower()]
        print("back buttons:", [b.text() for b in back])
        if back:
            ex.click(back[0])
            print("after back: mode", ex.w.mode, "sel", ex.s.sel_step)
            if ex.w.mode != 0:
                pp.note_once("back_graph", "gênant", "« Aller au graphe » ne revient pas au graphe", "", st)
    # the top mode button
    ex.click(ex.w.mode_btns[2])
    print("mode button Post:", ex.w.mode, post_page().pages.currentIndex() if post_page() else None)
    pp.activate()
    pp.key(Qt.Key_1, ex.w, Qt.ControlModifier)
    print("Ctrl+1 ->", ex.w.mode)
    if end == "stars":
        # Open in Post of a Combine block
        cb = pp.blocks_of("composition")[0]
        m = ex.right_click_block(cb)
        en = [d for d in m if "Ouvrir" in (d.get("text") or "")]
        ex.right_click_block(cb, "Ouvrir dans")
        print("open in post: mode", ex.w.mode, "page", post_page().pages.currentIndex())
        if post_page().pages.currentIndex() == 0 and en and en[0]["enabled"]:
            pp.note_once("open_in_post_empty", "mineur", "« Ouvrir dans Post-traitement » d'un bloc Combiner ouvre une "
                         "page vide", "Le menu du bloc Combiner offre « Ouvrir dans Post-traitement » (actif) alors que "
                         "rien n'est calculé : la page affiche « Pas encore de calque », le calque du bloc n'y est "
                         "pas.", st + ["Clic droit sur Combiner les calques étoiles et fond", "Ouvrir dans Post-traitement"])
        ex.click(ex.w.mode_btns[0])
    pp.check_slot_errors("page Post-traitement sans calcul", st)

# ---------------------------------------------------------------- 2. simulated: the workflows marked computed
print("=" * 20, "simulated computed")
session_project(ex)
build_chain(ex, CHAINS["stars"])
st = steps_chain("stars") + ["(état simulé : workflows marqués calculés, sans pixels)"]
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
pp.pull_port(pp.block("couleur"))                       # a middle block's image too
ex.settle(6)
pp.dump("simulated before Post")
ex.click_block(pp.final())
print("post panel:", pp.panel_texts(), pp.panel_buttons())
ex.click(ex.w.mode_btns[2])
ex.settle(20)
pg = post_page()
print("page index:", pg.pages.currentIndex())
v = pg.view
names = [(lay["kind"], lay.get("part"), lay["name"]) for lay in v.layers]
print("post layers:", names)
row_names = [r.n.text() for r in v.rows]
print("rows:", row_names)
pp.check_texts("calques de la page Post-traitement", row_names + visible_texts(v.right), st + ["Page Post-traitement"])
n_graph = int(pp.final_summary().split()[0])
n_rows = len([r for r in v.rows if r.lay["kind"] in ("workflow", "image")])
print("graph count", n_graph, "post rows", n_rows)
if n_graph != n_rows:
    pp.note_once("post_rows_count", "gênant", "Nombre de calques différent entre le bloc et la page Post-traitement",
                 f"Bloc : {n_graph} ; page : {n_rows} ({row_names}).", st)
ex.shot("post_page_simulated")
# select each workflow layer: its properties (Channels)
for r in list(v.rows):
    if r.lay["kind"] != "workflow":
        continue
    lid = r.lay["id"]
    v.select(lid, "layer")
    ex.settle(4)
    props = visible_texts(v.right)
    combos = [c for c in v.right.findChildren(QComboBox) if c.isVisible()]
    print("layer", r.lay["name"], "props:", props[:25], "combos:", [[c.itemText(i) for i in range(c.count())] for c in combos])
    pp.check_texts("propriétés d'un calque", props, st + [f"Sélectionner le calque {r.lay['name']}"])
    ch = [c for c in combos if c.findText("Lié") >= 0 or c.findText("Liés") >= 0 or c.count() == 2]
    print("channels combos:", [(c.currentText(), c.toolTip()) for c in ch])
v.select(next(lay["id"] for lay in v.layers if lay["kind"] == "workflow"), "layer")
ex.settle(4)
combos = [c for c in v.right.findChildren(QComboBox) if c.isVisible() and c.count() == 2]
if combos:
    c = combos[0]
    before = c.currentIndex()
    undo_before = (s.can_undo(), len(v.undo))
    c.setCurrentIndex(1 - before)
    ex.settle(6)
    pp.check_slot_errors("Canaux d'un calque de workflow changé", st)
    print("channels changed:", c.currentText(), "graph undo", s.can_undo(), "post undo", len(v.undo), undo_before)
    pp.undo()                                              # in Post: Post's own undo
    ex.settle(6)
    combos2 = [x for x in v.right.findChildren(QComboBox) if x.isVisible() and x.count() == 2]
    print("after Ctrl+Z in Post:", [x.currentText() for x in combos2])
    if combos2 and combos2[0].currentIndex() != before:
        pp.note_once("channels_undo", "gênant", "Ctrl+Z dans le Post-traitement n'annule pas le réglage « Canaux »",
                     "Le réglage Canaux (Liés / Chaque canal) d'un calque de workflow, changé dans la page "
                     "Post-traitement, reste après Ctrl+Z.", st + ["Page Post-traitement", "Canaux : changer", "Ctrl+Z"])
# layer menu
r = next(r for r in v.rows if r.lay["kind"] == "workflow")
ex.menu_pick = None
n = len(ex.menus)
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
QTest.mouseClick(r, Qt.RightButton, Qt.NoModifier, QPoint(30, 10))
ex.settle()
print("layer menu:", ex.menus[-1] if len(ex.menus) > n else None)
if len(ex.menus) > n:
    pp.check_texts("menu d'un calque", pp.menu_texts(ex.menus[-1]), st)
# delete the "Sortie de Couleur" layer in Post, then go back to the graph
tgt = next((lay for lay in v.layers if lay["kind"] == "workflow" and "Couleur" in lay["name"]), None)
if tgt is not None:
    v.select(tgt["id"], "layer")
    ex.settle(3)
    ex.click(v.b_del)
    ex.settle(10)
    pp.check_slot_errors("Supprimer un calque dans le Post-traitement", st)
    print("post layers after delete:", [lay["name"] for lay in v.layers])
    ex.click(ex.w.mode_btns[0])
    ex.settle(10)
    pp.dump("graph after a layer deleted in Post")
    wl = [w for w in pp.wires() if "Couleur" in w.toolTip()]
    if wl:
        pp.note_once("post_delete_wire_back", "gênant", "Calque supprimé dans le Post-traitement : son fil reste dans "
                     "le graphe", f"Fils : {[w.toolTip() for w in pp.wires()]}", st + ["Supprimer le calque dans Post"])
    br = next((q for q in s.paths if "Couleur" in q.name), None)
    if br is not None and not wl:
        pp.note_once("post_delete_branch", "question", "Calque supprimé dans le Post-traitement : la branche « Sortie de "
                     "Couleur » reste dans le graphe, sans fil", "Le calque n'est plus dans le Post, le workflow qui le "
                     "faisait reste (et sera calculé).", st + ["Envoyer Couleur au Post", "Le supprimer dans la page Post"])
    # back in Post: does the deleted layer come back?
    ex.click(ex.w.mode_btns[2])
    ex.settle(10)
    print("post layers after round trip:", [lay["name"] for lay in v.layers])
    if any("Couleur" in lay["name"] for lay in v.layers):
        pp.note_once("post_delete_comes_back", "gênant", "Un calque supprimé dans le Post-traitement revient",
                     "Après un aller-retour graphe / Post-traitement, le calque supprimé est de nouveau là.", st)
    # the graph's Ctrl+Z while Post is shown: Post's own undo
    pp.undo()
    ex.settle(6)
    print("post layers after Ctrl+Z:", [lay["name"] for lay in v.layers])
    ex.click(ex.w.mode_btns[0])
    ex.settle(10)
    pp.dump("graph after Ctrl+Z in Post")
# rename a layer in Post: the graph's name?
ex.click(ex.w.mode_btns[2])
ex.settle(6)
lay = next(lay for lay in v.layers if lay["kind"] == "workflow")
ex.dialog_handler = None
v.rename(lay, "Mon calque")
ex.settle(6)
ex.click(ex.w.mode_btns[0])
ex.settle(6)
print("after rename in Post: graph layers", pp.graph_layers(), "paths", [q.name for q in s.paths])
ex.click_block(pp.final())
print("post panel:", pp.panel_texts())
pp.check_slot_errors("aller-retour Post-traitement / graphe", st)
ex.close()
