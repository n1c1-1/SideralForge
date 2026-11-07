"""post_wires A: the wire of a workflow's image to Post, for chains ending at each block: drawn, hover, click,
Del, its menu, Ctrl+Z / Ctrl+Y, given back from the block's menu; the chain's wires not selectable."""
import sys

from explore_post_common import CHAINS, FR, VIEWKEY, P, Explorer, build_chain, session_project, start, steps_chain
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

start(1500)
ex = Explorer("post_wires")
pp = P(ex)
ends = sys.argv[1:] or ["stacking", "gradient", "denoise", "color", "stars"]

for end in ends:
    print("=" * 30, end)
    session_project(ex)
    build_chain(ex, CHAINS[end])
    pp.check_slot_errors(f"construction de la chaîne jusqu'à {FR[end]}", steps_chain(end))
    pp.dump("built")
    st = steps_chain(end)
    last = pp.block(VIEWKEY[end])
    ws = pp.wires()
    n_layers0 = len(pp.graph_layers())
    pp.check_texts("graphe", ex.texts(), st)
    if end == "stars":
        comp = pp.blocks_of("composition")
        print("compositions:", [(c.title, c.summary) for c in comp], "wires:", len(ws))
        # the Output is in the Split block: no wire of its image; the Combine block's wire is not removable
        if not ws and not comp:
            pp.note_once("stars_nothing", "gênant", "Split étoiles : rien ne va au Post-traitement",
                         "Aucun fil ni bloc Combiner après Split étoiles et fond.", st)
        # its menu: "Envoyer son image au Post" sends the Output image (before the split) too
        m = ex.right_click_block(last)
        print("menu last:", [d.get("text") for d in m])
        pp.check_texts("menu du dernier bloc", pp.menu_texts(m), st)
        continue
    if len(ws) != 1:
        pp.note_once(f"nwires_{end}", "gênant", f"Chaîne jusqu'à {FR[end]} : {len(ws)} fils vers le Post-traitement",
                     f"Attendu un fil amovible ; fils : {[w.toolTip() for w in ws]}", st)
        continue
    w = ws[0]
    pt = pp.free_point_on(w)
    if pt is None:
        pp.note_once(f"wire_hidden_{end}", "gênant", f"Fil vers le Post-traitement inatteignable ({FR[end]})",
                     "Aucun point du fil n'est cliquable : il est entièrement sous des blocs.", st)
        continue
    # hover
    vp = ex.view.viewport()
    c0 = w.pen().color().name()
    ex.hover(vp, pt)
    c1 = w.pen().color().name()
    print("hover colours", c0, c1, "cursor", w.cursor().shape())
    if c0 == c1:
        pp.note_once("hover", "mineur", "Le fil vers le Post-traitement ne s'allume pas au survol",
                     f"Couleur avant {c0}, au survol {c1}.", st)
    # a chain wire (not removable): clicking it selects nothing
    plain = [it for it in pp.plain_paths() if it.zValue() == 1]
    for it in plain[:3]:
        ptp = pp.free_point_on(it)
        if ptp is None:
            continue
        sel_before = ex.s.sel_step
        ex.click(vp, ptp)
        hit = ex.view.itemAt(ptp)
        print("chain wire click: sel", sel_before, "->", ex.s.sel_step, "sel_wire", getattr(ex.view, "sel_wire", None),
              "cursor", it.cursor().shape() if it.hasCursor() else None)
        if getattr(ex.view, "sel_wire", None) is not None and ex.view.sel_wire is not w:
            pp.note_once("plain_selectable", "gênant", "Un fil non amovible se sélectionne",
                         "Clic sur un fil de la chaîne : il est retenu comme fil sélectionné.", st)
        break
    # click selects
    pp.click_wire(w)
    if not w.selected:
        pp.note_once("click_sel", "gênant", "Clic sur le fil vers le Post-traitement : pas de sélection",
                     "Le fil ne passe pas à l'état sélectionné.", st)
    # selecting the wire: does the block selection go? (the panel still shows the block?)
    print("after wire click: sel_step", ex.s.sel_step, "panel head:", pp.panel_texts()[:2])
    if pp.once("wire_sel_panel") and ex.s.sel_step not in (None, "final"):
        blk_sel = [it for it in ex.blocks() if it.selected]
        if blk_sel:
            pp.note_once("wire_and_block_selected", "mineur",
                         "Fil sélectionné, mais le bloc précédent reste sélectionné (panneau et cadre)",
                         f"Après un clic sur le fil vers le Post-traitement, le bloc « {blk_sel[0].title} » garde son "
                         "cadre de sélection et le panneau de droite montre toujours ce bloc : on ne voit pas ce que "
                         "Suppr va retirer (le fil ou le bloc).", st + ["Clic sur le fil vers Post-traitement"])
    ex.shot(f"wire_selected_{end}")
    # Del removes it
    pp.del_key()
    pp.check_slot_errors("Suppr sur le fil vers le Post-traitement", st)
    pp.dump("after Del")
    toast = pp.toast()
    print("toast:", toast, "popups:", ex.popups[-2:])
    if pp.wires():
        pp.note_once(f"del_{end}", "gênant", "Suppr ne retire pas le fil vers le Post-traitement",
                     f"Chaîne jusqu'à {FR[end]} : après clic sur le fil et Suppr, le fil est toujours là.", st)
        continue
    if "Ctrl+Z" not in toast:
        pp.note_once("toast_no_undo", "mineur", "Retrait d'un fil : le message n'offre pas Ctrl+Z",
                     f"Message affiché : « {toast} ». Les autres retraits disent « Ctrl+Z annule ».",
                     st + ["Clic sur le fil vers le Post-traitement", "Suppr"])
    n_after = len(pp.graph_layers())
    summ = pp.final_summary()
    print("layers", n_layers0, "->", n_after, summ)
    # panel of Post block
    ex.click_block(pp.final())
    print("post panel:", pp.panel_texts())
    # Ctrl+Z: the wire comes back?
    paths_before = [(q.pid, [n['block'] for n in ex.s.p.workflow(q.pid).blocks]) for q in ex.s.paths]
    pp.undo()
    pp.check_slot_errors("Ctrl+Z après retrait du fil", st)
    paths_after = [(q.pid, [n['block'] for n in ex.s.p.workflow(q.pid).blocks]) for q in ex.s.paths]
    back = bool(pp.wires())
    print("undo: wire back", back, "paths", paths_before, "->", paths_after)
    if not back:
        changed = paths_before != paths_after
        pp.note_once("undo_wire", "gênant", "Ctrl+Z ne remet pas le fil retiré vers le Post-traitement" +
                     (" ; il annule à la place la modification précédente du graphe" if changed else ""),
                     f"Chaîne jusqu'à {FR[end]} : clic sur le fil vers Post-traitement, Suppr (le calque part), puis "
                     f"Ctrl+Z. Le fil ne revient pas." + (f" À la place, Ctrl+Z défait l'ajout précédent : le "
                                                          f"workflow passe de {paths_before[0][1]} à "
                                                          f"{paths_after[0][1]}." if changed else ""),
                     st + ["Clic sur le fil vers Post-traitement", "Suppr", "Ctrl+Z"])
        pp.redo()
    else:
        pp.redo()
        if pp.wires():
            pp.note_once("redo_wire", "gênant", "Ctrl+Y ne retire pas à nouveau le fil", "", st)
    # given back from the block's menu
    last = pp.block(VIEWKEY[end])
    m = ex.right_click_block(last)
    texts = [d.get("text") for d in m]
    print("menu last after removal:", texts)
    pp.check_texts("menu du dernier bloc", pp.menu_texts(m), st)
    if not any("Post" in (t or "") and "nvoyer" in (t or "") for t in texts):
        pp.note_once(f"no_send_back_{end}", "gênant", "Fil retiré : le menu du bloc n'offre pas de le renvoyer au "
                     "Post-traitement",
                     f"Le message dit « renvoyez-le depuis le menu de son bloc », mais le menu de {FR[end]} n'a pas "
                     f"d'entrée « Envoyer… ». Menu : {texts}", st + ["Retirer le fil vers Post-traitement",
                                                                    f"Clic droit sur {FR[end]}"])
        # does Ctrl+Z / undo remain the only way? try the Post block's menu
        mf = ex.right_click_block(pp.final())
        print("menu Post block:", [d.get("text") for d in mf])
        continue
    ex.right_click_block(last, "Vers Post")
    pp.check_slot_errors("Envoyer son image au Post (après retrait du fil)", st)
    pp.dump("after send back")
    if not pp.wires():
        pp.note_once(f"send_back_fail_{end}", "gênant", "« Envoyer son image au Post » ne remet pas le fil", "", st)
        continue
    # Ctrl+Z of the send-back
    pp.undo()
    print("undo of send back: wires", len(pp.wires()))
    if pp.wires():
        pp.note_once("undo_sendback", "gênant", "Ctrl+Z n'annule pas « Envoyer son image au Post »",
                     "Après avoir renvoyé l'image au Post-traitement depuis le menu du bloc, Ctrl+Z laisse le fil.",
                     st + ["Retirer le fil", "Clic droit sur le dernier bloc > Envoyer son image au Post", "Ctrl+Z"])
    pp.redo()
    # the wire's own menu: Remove
    w = pp.wires()[0] if pp.wires() else None
    if w is not None:
        m = pp.right_click_wire(w, "Retirer")
        pp.check_slot_errors("menu du fil vers Post-traitement > Retirer", st + ["Clic droit sur le fil vers "
                                                                                "Post-traitement", "Retirer"])
        print("wire menu", m, "wires after:", len(pp.wires()))
        if m and pp.wires():
            pp.note_once("wire_menu_remove", "gênant", "Menu du fil vers le Post-traitement : « Retirer » ne fait rien",
                         "Clic droit sur le fil, « Retirer » : le fil et son calque restent (une exception Python est "
                         "levée : le menu passe un booléen à la place du workflow).",
                         st + ["Clic droit sur le fil vers Post-traitement", "Retirer"])
    print("popups:", ex.popups, "modals:", ex.modals)
ex.close()
