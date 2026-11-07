"""post_wires E: once the composition is edited (a wire removed, an image of Split sent), a middle block's image sent
to Post: does it become a layer? Also a branch made by a palette drop under a block, and a second flow."""
import sys

from explore_post_common import CHAINS, FR, VIEWKEY, P, Explorer, build_chain, session_project, start, steps_chain

start(1800)
ex = Explorer("post_wires")
pp = P(ex)


def count():
    return pp.final_summary(), len(pp.graph_layers()), len(pp.wires())


for how in ("menu", "port"):
    act = ""
    # A: chain to Color, the wire removed then sent back: then Gradient's image sent
    print("=" * 20, "A", how)
    session_project(ex)
    build_chain(ex, CHAINS["color"])
    st = steps_chain("color")
    w = pp.wires()[0]
    pp.click_wire(w)
    pp.del_key()
    ex.right_click_block(pp.block("couleur"), "Vers Post")
    print("after remove + send back:", count(), (ex.s.p.post or {}).get("edited"))
    c0 = count()
    if how == "menu":
        ex.right_click_block(pp.block("fond"), "Vers Post")
    else:
        pp.pull_port(pp.block("fond"))
    ex.settle(6)
    pp.check_slot_errors("Envoyer Gradient au Post (composition modifiée)", st)
    c1 = count()
    print("after send gradient:", c1, [q.name for q in ex.s.paths], "toast", pp.toast())
    act = "Clic droit sur Gradient > Envoyer son image au Post" if how == "menu" else \
        "Tirer le port ● de Gradient sur Post-traitement"
    if len(ex.s.paths) == 2 and c1[1] == c0[1]:
        pp.note_once("edited_send_lost", "bloquant",
                     "Après un retrait de fil, « Envoyer son image au Post » d'un bloc ne fait plus de calque",
                     f"Le message dit « {pp.toast()} » et une branche « {ex.s.paths[-1].name} » apparaît dans le "
                     f"graphe, mais le bloc Post-traitement reste à « {c1[0]} » : aucun fil, aucun calque. Dès que la "
                     "composition a été modifiée une fois (un fil retiré puis renvoyé), les images envoyées depuis un "
                     "bloc du milieu ne vont plus au Post-traitement.",
                     st + ["Clic sur le fil vers Post-traitement, Suppr", "Clic droit sur Couleur > Envoyer son image au "
                                                                         "Post", act])
    ex.shot(f"edited_send_{how}")
    # B: chain to Split stars, its starless image sent, then Color's image
    print("=" * 20, "B", how)
    session_project(ex)
    build_chain(ex, CHAINS["stars"])
    st = steps_chain("stars")
    pp.pull_port(pp.block("etoiles"), "starless")
    c0 = count()
    if how == "menu":
        ex.right_click_block(pp.block("couleur"), "Vers Post")
    else:
        pp.pull_port(pp.block("couleur"))
    ex.settle(6)
    c1 = count()
    print("after send color:", c0, "->", c1, [q.name for q in ex.s.paths], "toast", pp.toast())
    if len(ex.s.paths) == 2 and c1[1] == c0[1]:
        pp.note_once("edited_send_lost_stars", "bloquant",
                     "Après l'envoi d'une image de Split au Post, l'image d'un autre bloc envoyée au Post est perdue",
                     f"Split étoiles : l'image sans étoiles tirée sur Post-traitement, puis {act.replace('Gradient', 'Couleur')}. "
                     f"Le message annonce le calque, une branche « {ex.s.paths[-1].name} » apparaît, mais le bloc "
                     f"Post-traitement reste à « {c1[0]} » sans fil pour elle.",
                     st + ["Tirer le ● du haut de Split sur Post-traitement", act.replace("Gradient", "Couleur")])
        ex.shot(f"edited_send_stars_{how}")
        # its Output block, and its menu: can it be sent?
        po = [b for b in ex.blocks() if b.node.key == "post"]
        if po:
            m = ex.right_click_block(po[0])
            print("branch Output menu:", [d.get("text") for d in m])

# C: a branch from a palette drop under a block (Denoise under Gradient), in a fresh and in an edited composition
for edited in (False, True):
    print("=" * 20, "C branch edited", edited)
    session_project(ex)
    build_chain(ex, CHAINS["gradient"])
    st = steps_chain("gradient")
    if edited:
        pp.click_wire(pp.wires()[0])
        pp.del_key()
        ex.right_click_block(pp.block("fond"), "Vers Post")
    c0 = count()
    ex.drag_palette_hover("denoise", ex.view.viewport().rect().center())
    gh = [g for g in ex.ghosts() if getattr(g, "branch_key", None)]
    print("branch ghosts:", [(g.text, g.branch_key) for g in gh])
    centres = {g.branch_key: ex.ghost_center(g) for g in gh}
    ex.drag_leave()
    key = "empilement" if "empilement" in centres else (list(centres)[0] if centres else None)
    if key is None:
        print("no branch ghost")
        continue
    ex.drag_palette("denoise", centres[key])
    pp.check_slot_errors("branche par glisser-déposer", st)
    c1 = count()
    print("after branch:", c0, "->", c1, [q.name for q in ex.s.paths], "toast", pp.toast())
    pp.dump("branch")
    if "précède Empilement" in pp.toast():
        pp.note_once("branch_toast_words", "mineur", "Message de branche : « ce qui précède Empilement est partagé »",
                     f"Message : « {pp.toast()} ». La branche part sous Empilement : Empilement lui-même est partagé "
                     "aussi (« jusqu'à Empilement » serait juste).",
                     st + ["Glisser Débruitage sur « nouvelle branche » sous Empilement"])
    if len(ex.s.paths) == 2 and c1[1] == c0[1]:
        pp.note_once(f"branch_not_post_{edited}", "bloquant" if edited else "gênant",
                     "Une nouvelle branche ne va pas au Post-traitement" + (" (composition déjà modifiée)" if edited else ""),
                     f"Débruitage glissé sur « nouvelle branche » sous Empilement : la branche « {ex.s.paths[-1].name} » "
                     f"est créée (son infobulle dit « elle va au Post comme calque à part »), mais le bloc "
                     f"Post-traitement reste à « {c1[0]} », sans fil.",
                     st + (["Retirer le fil vers Post-traitement puis le renvoyer (menu du bloc)"] if edited else []) +
                     ["Glisser Débruitage de la palette sur « nouvelle branche » sous Empilement"])
    pp.undo()
    print("undo branch:", count(), [q.name for q in ex.s.paths])
ex.close()
