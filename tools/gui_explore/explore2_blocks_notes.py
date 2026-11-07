"""block_actions round 2: the new anomalies, each reproduced twice by explore2_blocks_dig.py / explore2_blocks_seq.py /
explore2_blocks_matrix.py, noted with the screenshot taken when it showed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import Explorer  # noqa: E402

ex = Explorer("block_actions")
D = ex.out
R = "\n(reproduit 2 fois)"

NOTES = [
    ("bloquant", "« Supprimer workflow » sur un canal de Fusion : accepté, puis Ctrl+Y lève une erreur Python et ne refait rien",
     "Projet mono (sessions Ha et OIII), Astrométrie posée dans chacune, puis Fusion par double-clic (elle ajoute "
     "d'elle-même Débruitage et Étoiles/fond dans chaque canal). Clic droit sur Étoiles/fond du workflow Ha → "
     "« Supprimer workflow » : accepté sans refus (« Workflow « Ha » supprimé · Ctrl+Z »). La Fusion reste dessinée "
     "« HOO · Ha, OIII », entourée de rouge ; son infobulle dit « À jour » (rien n'a été calculé) et « Ha : sans "
     "workflow ». Ctrl+Z rend Ha. Ctrl+Y : erreur Python « ValueError: workflow 3: its fusion reads workflows the "
     "project does not have », et rien n'est refait. Même chose pour OIII. C'est le chemin « Supprimer le workflow » de "
     "l'anomalie bloquante de la première passe : la Session et l'Astrométrie ne plantent plus, ce menu-là si.",
     ["Projet mono Ha + OIII, Astrométrie dans chaque, double-clic Fusion",
      "Clic droit sur Étoiles/fond de Ha → Supprimer workflow", "Ctrl+Z puis Ctrl+Y"],
     D / "dig_fusion_wf_delete_0.png"),
    ("bloquant", "Bouton « Supprimer » du panneau de la Fusion : erreur Python",
     "Projet mono Ha + OIII, Astrométrie dans chaque, Fusion. Cliquer la Fusion, puis le bouton « Supprimer » de son "
     "panneau : la Fusion est bien retirée (« Fusion et suivants retirés · Ctrl+Z »), mais deux erreurs Python suivent "
     "(« KeyError: 'no workflow 3' », venues des champs de poids des canaux du panneau qui disparaît). Dans la vraie "
     "fenêtre ces erreurs s'affichent. Suppr ou « Supprimer » du menu sur la même Fusion : pas d'erreur.",
     ["Projet mono, Astrométrie dans Ha et OIII, Fusion", "Cliquer Fusion", "Panneau → Supprimer"],
     D / "dig_fusion_panel_delete_0.png"),
    ("gênant", "Calibration d'une session : « Pause vérif » annonce une pause qui n'est pas posée",
     "Deux sessions, Calibration → Alignement → Empilement → Astrométrie → Gradient. Clic droit sur la Calibration de "
     "Session 1 → « Pause vérif » : le message dit « Pause vérif après Calibration », mais aucune pause n'est dessinée, "
     "rien n'est enregistré et il n'y a rien à annuler. Même chose pour la Calibration de Session 2 (dans un projet à "
     "deux flux, où chaque session a son workflow, la pause est bien posée). Le panneau de cette Calibration dit aussi « À jour » alors que rien n'a été calculé.",
     ["Deux sessions + chaîne", "Clic droit sur la Calibration de Session 1 → Pause vérif"],
     D / "dig_session_calib_pause_1.png"),
    ("gênant", "Calibration d'une session : D et Suppr agissent sur les deux sessions, sans que ça se voie",
     "Deux sessions, chacune avec son bloc Calibration. Cliquer la Calibration de Session 1, touche D : le message dit "
     "« Calibration off · Workflow · Ctrl+Z » ; la Calibration des deux sessions est désactivée, mais aucun des deux "
     "blocs n'a l'air désactivé (même aspect qu'avant). Suppr sur la Calibration de Session 2 retire les deux blocs "
     "Calibration. La portée (quelle session) n'est dite nulle part. Le panneau de ces blocs n'a toujours ni « Supprimer » "
     "ni « Désactiver ».",
     ["Deux sessions + chaîne", "Cliquer la Calibration de Session 1, D", "Ctrl+Z, cliquer celle de Session 2, Suppr"],
     D / "dig_session_calib_off_1.png"),
    ("gênant", "Supprimer le dernier bloc d'une branche : la branche reste, un bloc « Sortie » prend sa place",
     "Chaîne jusqu'à Gradient, puis Couleur glissé sur « nouvelle branche » sous Astrométrie (workflows « Workflow » et "
     "« Workflow · Couleur »). Suppr sur ce Couleur : message « Couleur retiré · Ctrl+Z », mais le workflow "
     "« Workflow · Couleur » reste, un bloc « Sortie » est dessiné à la place de Couleur et Post compte toujours 2 calques "
     "(l'image d'Astrométrie part au Post). Le panneau d'Astrométrie propose encore « « Workflow · Couleur » seulement ». "
     "Même chose en supprimant Gradient sur la ligne principale. Ctrl+Z et Ctrl+Y marchent.",
     ["Chaîne jusqu'à Gradient, branche Couleur sous Astrométrie", "Cliquer Couleur de la branche, Suppr"],
     D / "seq_branch_del_1.png"),
    ("mineur", "Messages en français : « off » et accords au masculin",
     "Désactiver un bloc affiche « Calibration off · Workflow · Ctrl+Z » (mot anglais « off » ; avant : « désactivé »). "
     "Les autres messages accordent au masculin des noms féminins : « Calibration retiré · Ctrl+Z », « Astrométrie "
     "retiré », « Astrométrie gardé : requis par Couleur », « Sortie Gradient gardé : absent du workflow ». Aussi "
     "« Empilement gardé : suite requiert image ».",
     ["Cliquer Calibration, D", "Cliquer Calibration, Suppr", "Cliquer Astrométrie (Couleur après), Suppr"],
     D / "dig_session_calib_off_0.png"),
    ("mineur", "Session seule : Suppr et glisser vers la palette ne disent rien",
     "Projet à une session. Son menu dit « Supprimer — seule session » (grisé, avec la raison). Mais Suppr sur le bloc "
     "Session 1, ou le glisser sur la palette, ne fait rien et n'affiche aucun message. Même silence pour D sur une "
     "Session, une Fusion ou un Recombiner.",
     ["Projet à une session + chaîne", "Cliquer Session 1, Suppr", "Glisser Session 1 sur la palette"],
     D / "seq_lone_session_del_1.png"),
    ("mineur", "Noms en double acceptés : deux sessions ou deux workflows du même nom, menus ambigus",
     "Renommer Session 1 en « Session 2 » dans son panneau : accepté, deux blocs « Session 2 ». Renommer la branche "
     "« Workflow · Couleur » en « Workflow » (menu Renommer) : accepté ; le menu de Calibration dit alors « Retirer de "
     "Workflow » deux fois et « Désactiver dans Workflow » deux fois, et le panneau « « Workflow » seulement » deux fois : "
     "on ne sait plus lequel est lequel. Aucun message. (Un nom vide est ignoré sans un mot.)",
     ["Chaîne jusqu'à Gradient, branche Couleur sous Astrométrie", "Clic droit Couleur → Renommer → « Workflow »",
      "Clic droit Calibration"],
     D / "dig_dup_names_1.png"),
    ("mineur", "Retirer la Session d'un canal de Fusion : une Session vide la remplace et les fils se croisent",
     "Projet mono Ha + OIII avec Fusion. La croix de la Session Ha la retire (« « Ha » retirée · fichiers gardés · "
     "Ctrl+Z »), plus d'erreur Python. Mais un bloc Session « Ha » vide (« brutes à ajouter ») reste à sa place, l'ordre "
     "des sessions s'inverse (OIII en haut, Ha en bas) alors que les workflows restent dans l'ordre Ha puis OIII : les "
     "fils entre Calibration et Alignement se croisent. La Fusion est entourée de rouge et Exécuter est grisé (« HOO "
     "bloquée : Ha : sans brutes »).",
     ["Projet mono, Astrométrie dans chaque, Fusion", "Croix de la Session Ha"],
     D / "seq_mono_removed_1.png"),
    ("mineur", "Retirer une Session d'un projet à deux flux : son flux part aussi, le message ne le dit pas",
     "Projet à deux flux (Session 1 → Calibration → Alignement → Empilement, Session 2 de même). Croix, Suppr, menu ou "
     "glisser vers la palette sur Session 1 : tout son flux disparaît (Calibration, Alignement, Empilement), le message "
     "dit seulement « « Session 1 » retirée · fichiers gardés · Ctrl+Z ». Pour une Image, le message dit « Image et "
     "suivants retirés ».",
     ["Projet à deux flux, chaîne dans chaque", "Croix de Session 1"],
     D / "dig_two_flows_session_1.png"),
    ("question", "« Ouvrir dans Post » sur le bloc Recombiner crée une étape d'annulation et un calque au nom anglais",
     "Chaîne complète jusqu'à Étoiles/fond (le bloc Recombiner est donné d'office). Panneau du Recombiner → « Ouvrir "
     "dans Post » : Post s'ouvre et la composition enregistre un calque « Workflow · Recombine » (mot anglais). De "
     "retour au graphe, le premier Ctrl+Z annule cette ouverture (le calque disparaît de la composition), pas la "
     "dernière modification faite. Ouvrir n'est pas une modification : est-ce voulu ?",
     ["Chaîne complète", "Cliquer Recombiner → Ouvrir dans Post", "Revenir au Graphe, Ctrl+Z"],
     D / "dig_open_in_post_1.png"),
]

for sev, title, detail, steps, shot in NOTES:
    ex.note(sev, title, detail + R, shot=str(shot) if shot else False, steps=steps)
print(len(NOTES), "noted")
