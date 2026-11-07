"""post_wires round 2: writes recheck.jsonl from the replays (explore2_post_recheck_a/b/c/d, explore2_post_page)."""
import json
from pathlib import Path

OUT = Path("F:/astro_essais/A_REGARDER/exploration2/post_wires")
R1 = Path("F:/astro_essais/A_REGARDER/exploration/post_wires/anomalies.jsonl")
titles = [json.loads(x)["title"] for x in R1.read_text(encoding="utf-8").splitlines() if x.strip()]


def shot(name):
    f = OUT / f"{name}.png"
    return str(f) if f.is_file() else None


S = [  # (status, note, shot) in the order of round 1
    ("corrigé", "Le bouton « Ouvrir Post » grisé a maintenant une infobulle : « Aucun calque n'est encore calculé : "
     "exécutez d'abord un workflow ».", None),
    ("toujours là", "Clic droit sur le bloc Post : toujours aucun menu.", None),
    ("toujours là", "Le bloc dit « 1 calque », la page Post (double-clic sur le bloc) dit toujours « Pas encore de "
     "calque » sans nommer les calques attendus ; seul le lien est devenu « Graphe → ».", "recheck_03_post_page"),
    ("toujours là", "Après un clic sur le fil vers Post, Empilement garde son cadre et le panneau montre toujours "
     "Empilement.", "recheck_04_wire_selected"),
    ("toujours là", "Après « Vers Post » sur Gradient, un bloc « Sortie » (« Sortie Gradient · vers Post ») apparaît "
     "toujours entre Gradient et Post, avec son panneau (Étirement, Canaux liés).", "recheck_05_gradient_sent"),
    ("toujours là", "« Ouvrir dans Post » du bloc Recombiner est actif et ouvre la page vide « Pas encore de calque ».",
     "recheck_06_combine_open"),
    ("toujours là", "Menu du bloc Sortie : « Désactiver — absent du workflow », et « Supprimer workflow » puis "
     "« Supprimer » sans séparateur.", None),
    ("toujours là", "Le message du retrait du fil est « « Workflow » hors Post · menu : Vers Post » : il ne dit pas que "
     "Ctrl+Z annule (alors que Ctrl+Z marche).", "recheck_08_toast"),
    ("toujours là", "Suppr sur le bloc Sortie répond « Sortie Gradient gardé : absent du workflow » : le bloc est nommé "
     "comme le workflow et la raison est fausse ; le menu « Supprimer » répond pareil, le bloc n'a pas de croix, et le glisser sur la palette ne fait rien, sans message.", "recheck_09_del_output"),
    ("toujours là", "Panneau du bloc Sortie : une case « Canaux liés » ; calque dans la page Post : une liste « Canaux : "
     "Liés / Chaque canal ». C'est bien le même réglage (changer l'un change l'autre).", "recheck_10_output_panel"),
    ("corrigé", "Ctrl+Z remet le fil retiré (et seulement lui), Ctrl+Y le retire de nouveau.", "recheck_11_undo"),
    ("corrigé", "« Supprimer workflow » sur le bloc Sortie se fait tout de suite, avec « … supprimé · Ctrl+Z », sans "
     "fenêtre ; Ctrl+Z et Ctrl+Y marchent.", "recheck_12_delete_workflow"),
    ("corrigé", "Ctrl+Z annule le renvoi « Vers Post » (le fil repart), Ctrl+Y le refait.", "recheck_13_undo_sendback"),
    ("corrigé", "Menu du fil > « Retirer » : plus d'erreur Python.", "recheck_15_wire_menu"),
    ("corrigé", "Menu du fil > « Retirer » retire le fil et son calque.", "recheck_15_wire_menu"),
    ("corrigé", "Ctrl+Z remet le fil « Étoiles seules → Post » retiré, Ctrl+Y le retire.", "recheck_16_undo_stars_wire"),
    ("toujours là", "Page Post (état simulé) : Canaux passé à « Chaque canal », Ctrl+Z vide la pile d'annulation mais la "
     "liste reste sur « Chaque canal » ; le mode de fusion, lui, est bien annulé.", "post_undo_channels"),
    ("corrigé", "Ctrl+Z jusqu'à retirer Étoiles/fond : le bloc Post revient à « 1 calque » (Workflow) avec son fil.",
     "recheck_18_split_undone"),
    ("corrigé", "Partout « Post » : bloc, onglet, menus (« Vers Post », « Sans étoiles vers Post »), messages (« vers "
     "Post », « hors Post »).", None),
    ("corrigé", "Suppr sur Recombiner puis Ctrl+Z : le bloc revient avec son calque ; Ctrl+Y le retire.",
     "recheck_20_undo_combine"),
    ("toujours là", "Deux blocs Recombiner s'appellent tous deux « Recombiner », et le panneau Post liste deux fois "
     "« Workflow · Recombiner ».", "recheck_21_combine_names"),
    ("toujours là", "Menu du bloc Recombiner : « Supprimer » sans « Suppr » (les autres blocs : « Supprimer  Suppr »).",
     None),
    ("corrigé", "Croix d'un Recombiner puis Ctrl+Z : ce Recombiner revient, Étoiles/fond reste.",
     "recheck_23_undo_cross"),
    ("corrigé", "Étoiles/fond supprimé : le Post compte 1 calque (Workflow) avec son fil ; Ctrl+Z remet Étoiles/fond et "
     "son Recombiner.", "recheck_24_split_deleted"),
    ("corrigé", "Le message est maintenant « Calque « Empilement brut » (Empilement) vers Post » : plus d'élision "
     "fautive dans ce message (mais voir la nouvelle anomalie « Retirer de Empilement brut »).", "recheck_25_toast"),
    ("corrigé", "Fil de la branche envoyée retiré : la branche s'en va avec lui (« « Empilement brut » supprimé · "
     "Ctrl+Z ») ; Ctrl+Z la remet.", "recheck_26_branch_wire_removed"),
    ("corrigé", "« Retirer de 2 workflows » sur Empilement envoyé : refusé partout (« Empilement gardé : suite requiert "
     "image »), rien n'est retiré nulle part.", "recheck_27_shared_stacking"),
    ("toujours là", "Gradient envoyé puis « Retirer de 2 workflows » : la branche s'arrête à Astrométrie mais s'appelle "
     "toujours « Sortie Gradient » (calque et bloc Sortie).", "recheck_27_shared_gradient"),
    ("corrigé", "Fil retiré, puis Couleur et Gradient renvoyés : 2 calques, 2 fils.", "recheck_29_after_removal_send"),
    ("corrigé", "Image sans étoiles tirée sur Post, puis « Vers Post » sur Couleur : 3 calques, 3 fils.",
     "recheck_30_split_then_color"),
    ("corrigé", "Le message est maintenant « Nouvelle branche · Empilement · Ctrl+Z ».", "recheck_31_branch_toast"),
    ("corrigé", "Après un fil retiré puis renvoyé, la nouvelle branche Débruitage va au Post (2 calques).",
     "recheck_32_new_branch"),
    ("corrigé", "Un bloc Image sans fichier : « 0 calque », pas de fil (mais voir la nouvelle anomalie des deux blocs "
     "Sortie).", "recheck_33_image_nofile"),
    ("corrigé", "Session vide puis Calibration, Alignement, Empilement : « 0 calque », pas de fil.",
     "recheck_34_empty_session"),
    ("toujours là", "Bloc Image suivi d'Astrométrie, Gradient, Couleur : son menu n'a que « Supprimer », pas de port ● "
     "ni « Vers Post » (seul, ou suivi d'Étoiles/fond, il a « Vers Post »).", None),
    ("corrigé", "Deux flux, fil du flux 2 retiré, port d'Empilement du flux 1 tiré : 2 calques, 2 fils.",
     "recheck_36_two_flows"),
    ("toujours là", "Le dernier bloc d'un canal de Fusion offre toujours « Vers Post » ; le canal devient un calque à "
     "côté de HOO (nommé « Empilement brut », sans dire Ha).", "recheck_37_channel_sent"),
    ("corrigé", "Workflow renommé « M31 final » : les calques deviennent « M31 final · Image sans étoiles » et « M31 "
     "final · Recombiner » ; Ctrl+Z / Ctrl+Y suivent.", "recheck_38_renamed"),
    ("toujours là", "Bouton « Hors Post » du panneau d'Étoiles/fond : le fil part, aucun message ; Ctrl+Z marche mais "
     "n'est pas annoncé.", "recheck_39_panel_out"),
    ("toujours là", "Bouton « Vers Post » du panneau d'Étoiles/fond : le fil apparaît, aucun message.",
     "recheck_40_panel_send"),
    ("corrigé", "Supprimer un Recombiner (panneau ou Suppr) : « Recombiner retiré · Ctrl+Z ».",
     "recheck_41_panel_delete"),
]
assert len(S) == len(titles), (len(S), len(titles))
with (OUT / "recheck.jsonl").open("w", encoding="utf-8") as fh:
    for t, (status, note, sh) in zip(titles, S):
        fh.write(json.dumps({"title": t, "status": status, "note": note, "shot": shot(sh) if sh else None},
                            ensure_ascii=False) + "\n")
from collections import Counter
print(Counter(s for s, _, _ in S))
