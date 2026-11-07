"""post_wires round 3 (control): writes recheck.jsonl (one line per round-2 open point, judged from the replays of
explore3_post_a/b/c/d.py) and notes the new problems seen twice in those replays."""
import json

from harness import Explorer

ex = Explorer("post_wires")
D = ex.out


def s(name):
    f = D / f"{name}.png"
    return str(f) if f.is_file() else None


C, T, X = "corrigé", "toujours là", "changé"
REC = [
    # round-2 new points
    ("Calque supprimé dans la page Post : sa branche reste dans le graphe, sans fil", C,
     "Supprimer le calque « Workflow · Gradient » dans la page Post retire aussi sa branche du graphe (1 workflow, 1 "
     "calque) ; Ctrl+Z dans Post la remet.", "n1_layer_deleted_in_post_run2"),
    ("Fil d'une branche retiré puis Ctrl+Z : le calque revient sans ses réglages de Post", C,
     "Calque mis en Écran, fil de la branche retiré puis Ctrl+Z : le calque revient en Écran.", "n2_layer_settings_run2"),
    ("Deux blocs « Sortie … vers Post » alors que Post dit « 0 calque »", C,
     "Deux double-clics sur Image font deux blocs « Image 1 » et « Image 2 », sans fil ni bloc Sortie, et Post dit "
     "« 0 calque ».", "n3_two_images_run2"),
    ("Branche envoyée au Post prolongée par un bloc : le calque garde l'ancien nom", C,
     "Branche Gradient sélectionnée par son fil puis Débruitage ajouté : le calque et le workflow deviennent "
     "« Workflow · Débruitage ».", "n4_branch_grown_run2"),
    ("Calques envoyés depuis deux canaux : rien ne dit lequel est Ha", C,
     "« Vers Post » sur le Débruitage de Ha puis d'OIII donne les calques « Ha · Débruitage » et « OIII · Débruitage ».",
     "m_channel_denoise_run2"),
    ("Français : « Retirer de Empilement brut » au lieu de « Retirer d'Empilement brut »", C,
     "Le menu dit maintenant « Supprimer dans Workflow · Empilement brut » : plus d'élision à faire.", "n6_menu_run2"),
    ("Port tiré une deuxième fois sur Post : rien ne se passe, sans message", T,
     "Le port de Gradient tiré une deuxième fois sur Post ne fait toujours rien et n'affiche aucun message.",
     "n7_port_twice_run2"),
    ("« Ouvrir Post » grisé, mais le double-clic sur le bloc Post ouvre la page", C,
     "« Ouvrir Post » est actif (« Rien de calculé : Post liste les calques à venir ») et la page, par le bouton comme "
     "par le double-clic, liste les calques à venir.", "n8_dbl_post_run2"),
    ("1366×768 : le bloc Post est hors de vue avec une chaîne complète", T,
     "À 1366×768, chaîne jusqu'à Étoiles/fond : le graphe est à l'échelle 0,75 et Post reste hors de vue à droite "
     "(252 px à défiler).", "n9_1366_run2"),
    ("« Vers Post » qui rend son fil au workflow : aucun message", C,
     "Le renvoi dit maintenant « « Workflow » vers Post · Ctrl+Z ».", "n11_send_back_run2"),
    ("Plus aucun calque dans Post : le panneau est vide et Exécuter calcule quand même", T,
     "Fil retiré : Post dit « 0 calque », son panneau ne montre que « Post » et le bouton dit toujours « Exécuter · "
     "≈ 9 min », actif.", "n12_zero_layer_run2"),
    ("« + Recombiner » : aucun message", C,
     "« + Recombiner » dit maintenant « Calque « Workflow · Recombiner » (Recombiner) vers Post · Ctrl+Z ».",
     "n13_recombine_run2"),
    ("Fusion supprimée : les canaux gardent ce que la Fusion leur a ajouté", C,
     "La Fusion supprimée rend aux canaux exactement leurs blocs et leurs calques « Ha », « OIII » d'avant ; Ctrl+Z la "
     "remet avec Débruitage et Étoiles/fond.", "m_fusion_deleted_run2"),
    ("« Vers Post » d'Étoiles/fond : on ne sait pas quelle image part", X,
     "Le menu du bloc n'a plus que « Sans étoiles vers Post » et « Étoiles vers Post », mais son sous-menu « Workflow "
     "« Workflow » » a encore « Vers Post », qui fait un calque nommé seulement « Workflow » sans dire que c'est "
     "l'image d'avant le retrait des étoiles.", "n15_split_send_run2"),
    ("Fil sélectionné : un clic dans le vide ou Échap ne le désélectionne pas", C,
     "Un clic dans le vide et Échap désélectionnent le fil ; Suppr ne fait ensuite plus rien.", "n16_wire_select_run2"),
    ("Deux workflows peuvent porter le même nom : deux calques identiques dans Post", C,
     "Renommer la branche « Workflow » est refusé : « Nom refusé : « Workflow » : nom d'un autre workflow ».",
     "n17_same_name_run2"),
    ("Les images des blocs du milieu arrivent masquées dans Post", C,
     "Les calques envoyés depuis Gradient, Empilement et Étoiles/fond arrivent tous visibles.", "n18_visible_run2"),
    ("Calque renommé dans Post : le graphe garde l'ancien nom", T,
     "Le calque renommé « Fond seul » dans Post : le workflow et son fil s'appellent toujours « Workflow · Gradient ».",
     "n19_rename_in_post_run2"),
    ("Page Post : « Supprimer » grisé sur le seul calque, sans dire pourquoi", T,
     "Seul calque « Workflow » sélectionné : « Supprimer » est grisé, toujours sans infobulle.",
     "n20_delete_single_run2"),
    # round-1 points still open in round 2
    ("Le bloc Post-traitement n'a pas de menu contextuel", C, "Clic droit sur Post : un menu avec « Ouvrir Post ».",
     None),
    ("Le bloc Post-traitement annonce des calques, la page Post-traitement dit « Pas encore de calque »", C,
     "La page Post dit « Les calques que Post aura une fois leurs workflows calculés : » et nomme « Workflow · pas "
     "encore calculé ».", "r_post_page_run2"),
    ("Fil sélectionné, mais le bloc précédent reste sélectionné (panneau et cadre)", C,
     "Après un clic sur le fil, aucun bloc ne reste choisi (mais le panneau de droite est vide : noté à part).",
     "r_wire_selected_run2"),
    ("Un bloc « Sortie » apparaît dans le graphe après un envoi au Post", C,
     "Après « Vers Post » sur Gradient, plus aucun bloc Sortie : la branche va au Post par un fil.",
     "n7_port_twice_run2"),
    ("« Ouvrir dans Post-traitement » d'un bloc Combiner ouvre une page vide", C,
     "« Ouvrir Post » du Recombiner ouvre la page qui nomme « Workflow · Recombiner · pas encore calculé ».",
     "r_combine_open_run2"),
    ("Menu du bloc Sortie : « Désactiver — ce bloc n'est pas dans le workflow »", C,
     "Plus de bloc Sortie ; le menu du fil a Renommer, Dupliquer, Enregistrer modèle…, Retirer, Supprimer workflow, "
     "avec des séparateurs.", None),
    ("Retrait d'un fil : le message n'offre pas Ctrl+Z", C, "Le message est « « Workflow » hors Post · Ctrl+Z ».", None),
    ("Suppr sur le bloc Sortie : message incompréhensible", C,
     "Plus de bloc Sortie ; Suppr sur le fil d'une branche la supprime avec « « Workflow · Gradient » supprimé · "
     "Ctrl+Z ».", None),
    ("Le réglage des canaux de la Sortie n'a pas le même nom ni la même commande dans le graphe et dans le "
     "Post-traitement", C, "Le réglage n'existe plus qu'à un endroit : la case « Canaux liés » du calque dans Post.",
     "d_linked_clicked_run2"),
    ("Ctrl+Z dans le Post-traitement n'annule pas le réglage « Canaux »", C,
     "Case « Canaux liés » décochée dans Post, puis Ctrl+Z : elle est recochée.", "d_linked_undo_run2"),
    ("Plusieurs blocs Combiner portent le même nom", C,
     "Les deux blocs s'appellent « Recombiner » et « Recombiner 2 », leurs calques aussi.", "r_two_combines_run2"),
    ("Menu du bloc Combiner : « Supprimer le bloc » sans le raccourci Suppr", C,
     "Le menu du Recombiner a « Supprimer  Suppr ».", None),
    ("Le calque envoyé garde le nom d'un bloc qui n'est plus là", C,
     "Gradient envoyé puis « Supprimer dans 2 workflows » : la branche devient « Workflow · Astrométrie ».",
     "r_shared_remove_run2"),
    ("L'image d'un bloc Image ne peut pas être envoyée au Post-traitement telle quelle", C,
     "Bloc Image suivi d'Astrométrie, Gradient, Couleur : son menu a « Vers Post », qui fait le calque « Image telle "
     "quelle ».", "r_image_chain_run2"),
    ("Le menu d'un canal de Fusion offre « Envoyer son image au Post » : le canal devient alors un calque à part", X,
     "Le canal peut toujours être envoyé au Post et devient un calque à part, mais ce calque porte maintenant le nom "
     "du canal (« Ha », « Ha · Débruitage »).", "m_channels_send_run2"),
    ("Panneau de Split : « Retirer du Post » ne dit rien et n'offre pas Ctrl+Z", C,
     "« Hors Post » dit « Image sans étoiles de « Workflow » hors Post · Ctrl+Z » et Ctrl+Z remet le fil.",
     "c_split_out_fr1600_run2"),
    ("Panneau de Split : « Vers le Post » ne dit rien", C,
     "« Vers Post » dit « Image sans étoiles · « Workflow » vers Post · Ctrl+Z ».", "c_split_panel_fr1600_run2"),
]
with (D / "recheck.jsonl").open("w", encoding="utf-8") as fh:
    for title, status, note, shot in REC:
        fh.write(json.dumps({"title": title, "status": status, "note": note, "shot": s(shot) if shot else None},
                            ensure_ascii=False) + "\n")
print({k: sum(1 for r in REC if r[1] == k) for k in (C, T, X)}, len(REC))

ST = ["Nouveau projet avec une session de 15 brutes", "Double-clic dans la palette : Calibration … Couleur, "
      "Étoiles/fond"]
NEW = [
    ("mineur", "Fil vers Post sélectionné : le panneau de droite est vide",
     "Un clic sur un fil vers Post (celui du workflow, ou celui d'une image d'Étoiles/fond) l'allume et ne laisse plus "
     "aucun bloc choisi, mais le panneau de droite devient entièrement vide : ni titre, ni nom du calque, ni « Retirer » "
     "ou « Ouvrir Post ». Vu deux fois, sur le fil du workflow et sur celui de l'image sans étoiles.",
     ST + ["Clic droit sur Étoiles/fond > Sans étoiles vers Post", "Clic sur le fil de l'image sans étoiles"],
     "r_wire_panel_run2"),
    ("mineur", "Empilement envoyé au Post après une branche : calque nommé « Workflow · Gradient · Empilement brut »",
     "Gradient envoyé au Post (branche « Workflow · Gradient »), puis le port d'Empilement, bloc commun aux deux "
     "workflows, tiré sur Post : le nouveau calque s'appelle « Workflow · Gradient · Empilement brut », comme si "
     "l'image venait de la branche Gradient, alors que c'est l'empilement commun. Ensuite « Vers Post » dans le menu "
     "d'Empilement ne fait rien, sans message. Vu deux fois.",
     ["Nouveau projet avec une session de 15 brutes", "Double-clic dans la palette : Calibration … Couleur",
      "Clic droit sur Gradient > Vers Post", "Tirer le port ● d'Empilement sur Post"], "r_shared_pull_name_run2"),
    ("mineur", "Page Post : l'explication du calque « Étoiles seules » parle du « bloc Étoiles »",
     "Calque « Workflow · Étoiles seules » choisi dans la page Post : son explication dit « Les étoiles seules du bloc "
     "Étoiles, sur fond noir … », alors que le bloc s'appelle « Étoiles/fond » partout ailleurs. Vu deux fois.",
     ST + ["(état simulé : workflow calculé)", "Clic droit sur Étoiles/fond > Sans étoiles vers Post, puis Étoiles "
           "vers Post", "Onglet Post, choisir le calque « Workflow · Étoiles seules »"], "r_post_right_run2"),
    ("mineur", "« Vers Post » n'est pas à la même place dans les menus des blocs",
     "Empilement, Gradient, Débruitage : « Vers Post » vient après « Inspection », sans séparateur (« Supprimer, "
     "Désactiver, Inspection, Vers Post, Pause vérif »). Image, Fusion : « Vers Post » en premier. Étoiles/fond : "
     "« Sans étoiles vers Post », « Étoiles vers Post » en premier, et encore « Vers Post » dans le sous-menu "
     "« Workflow « Workflow » ». Dernier bloc (Couleur) : seulement dans ce sous-menu. Vu deux fois.",
     ["Nouveau projet avec une session de 15 brutes", "Double-clic dans la palette : Calibration … Couleur",
      "Clic droit sur Empilement, Gradient, Débruitage, Couleur ; puis sur un bloc Image et sur Fusion"], None),
]
for sev, title, detail, steps, shot in NEW:
    ex.note(sev, title, detail, shot=s(shot) if shot else False, steps=steps)
