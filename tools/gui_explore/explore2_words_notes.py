"""Words and layout, round 2: the new anomalies, written with ex.note after each was reproduced at least twice (two
sizes or two languages) by explore2_words_walk.py, explore2_words_recheck.py, explore2_words_undo.py,
explore2_words_errors.py, explore2_words_shots.py and explore2_words_compact.py. Appends to <out>/anomalies.jsonl
(run once).
"""
from __future__ import annotations

from harness import Explorer

ex = Explorer("words_layout", lang="fr", size=(1366, 768))
O = ex.out
S = O / "recheck"
N = O / "notes2"
W = O / "shots"

ex.note("gênant", "« off » : un mot anglais en français, à côté de « Désactivé » pour le même état",
        "Un bloc désactivé est dit « off » en français : message « Gradient 2 off · Workflow · Ctrl+Z » (ou « · 2 "
        "workflows »), Journal « Gradient 2 off · 2 workflows », résumé sous le titre du bloc dans le graphe « off : "
        "image transmise », infobulle du bloc « off : image transmise … Désactivé : l'image passe. D ou le menu le "
        "réactive. ». Le catalogue a aussi « {bloc} déjà là, off : réactiver (D) », « off : brutes telles quelles », "
        "« off : brutes déjà alignées ». Partout ailleurs (menu, bouton du panneau, Inspection) c'est « Désactiver » / "
        "« Désactivé » / « Réactiver ».",
        shot=str(W / "fr_1366x768" / "action_fond_Désactiver.png"),
        steps=["Français, chaîne complète avec Gradient placé deux fois", "Clic droit sur Gradient 2 > Désactiver",
               "Lire le message en bas, le résumé du bloc, Affichage > Journal"])

ex.note("gênant", "Projet d'une image à l'ouverture : la palette dit « Poser Session/Image d'abord » alors que l'Image est là",
        "À l'ouverture d'un projet dont la seule source est un bloc Image, rien n'est choisi : la palette dit « Blocs · "
        "nouveau workflow », tous les blocs d'image sont grisés avec la raison « Poser Session/Image d'abord » (en "
        "anglais « Place Session/Image first ») alors que le bloc Image est dessiné dans le graphe ; le panneau de "
        "droite montre « Images » avec un seul bouton « Importer dossier… » (un panneau de brutes, dans un projet "
        "sans session) ; la ligne d'état dit « aucun bloc : en poser un ». Il faut deviner qu'il faut d'abord cliquer "
        "sur le bloc Image. Vu aux deux tailles, dans les deux langues.",
        shot=str(N / "fr_raison_pas_de_alignement.png"),
        steps=["Ouvrir un projet dont le workflow part d'un fichier image", "Survoler Alignement ou Gradient dans la palette"])

ex.note("mineur", "« Sortie » nomme trois choses, et la raison d'Étoiles/fond parle d'un bloc absent de l'écran",
        "En survolant Étoiles/fond (grisé) dans un projet à session, la raison est « Sortie requiert Empilement avant » "
        "(« Output needs Stacking first ») : on survole Étoiles/fond et on lit le nom d'un autre bloc, qui n'est pas "
        "dessiné. « Sortie » désigne aussi : le petit rond de sortie des blocs (infobulle « Sortie ● : la glisser sur "
        "Post pour en faire un calque »), un bouton du bandeau de l'Inspection, et les blocs dessinés après « Vers "
        "Post » sur Empilement — deux blocs portent alors le même titre « Sortie » (« Empilement brut · vers Post » et "
        "« Empilement brut (copie) · vers Post ») ; leur menu a à la fois « Supprimer workflow » et « Supprimer ».",
        shot=str(N / "fr_raison_etoiles_sortie.png"),
        steps=["Projet à une session, placer Calibration", "Survoler Étoiles/fond dans la palette",
               "Clic droit sur Empilement > Vers Post, puis Dupliquer le workflow créé : deux blocs « Sortie »"])

ex.note("mineur", "Règle des noms courts (deux mots, sans article) : boutons, menus, cases et réglages encore longs",
        "Boutons et menus : « Enregistrer comme modèle… » (menu, bouton du panneau, titre de la fenêtre), « Importer "
        "un dossier comme nouveau projet… » (Accueil), « Télécharger les positions des étoiles (1,1 Go) », "
        "« Télécharger les spectres des étoiles (10,6 Go) », « Vider cache des projets fermés », « Toutes les sessions "
        "ensemble » (Inspection d'une session), « Jusqu'au bloc choisi » (menu Exécuter), « Ouvrir dans Post ». "
        "Cases : « Étirer automatiquement les images linéaires par défaut », « Afficher ce que le moteur doit encore "
        "construire (◇) », « Garder brutes calibrées et alignées ». Onglets : « À partir de brutes », « À partir d'une "
        "image », « Copie d'un workflow ». Noms de réglages avec article : « Degré de la surface », « Nom de la "
        "cible », « Taille de la goutte », « Mode de fusion des étoiles », « Seuil de détection des étoiles », "
        "« Balance des blancs de l'appareil », « Taille maximale des étoiles », « Seuil des défauts du flat », "
        "« Pixels chauds repérés dans les brutes », « % de la meilleure », « Âge maximal d'un dark », « Sections de "
        "l'accueil », « Onglet de la galerie ». Mêmes cas en anglais (« Import folder as new project… », « Clear "
        "cache of closed projects », « Auto-stretch linear images by default », « Maximum age of a dark »…). Liste "
        "complète : compact_names.txt. À juger : la règle vaut-elle aussi pour les noms de réglages ?",
        shot=str(W / "fr_1366x768" / "settings_3_catalogs.png"),
        steps=["Français : Accueil, Paramètres (chaque page), panneaux des blocs, menus des blocs"])

ex.note("mineur", "Une même chose, deux noms : Jusqu'ici / Jusqu'au bloc choisi, Supprimer / Retirer, Flux / workflow…",
        "« Jusqu'ici » (bouton de l'Inspection) et « Jusqu'au bloc choisi » (menu Exécuter) pour la même action ; titre de "
        "sa fenêtre « Exécuter « Workflow » jusqu'à Gradient 2 » (en anglais « Run to here » / « Run to selected »). "
        "« Supprimer » (menu, bouton du panneau) mais « Retirer ce bloc… » (infobulle du même bouton), « Gradient 1 "
        "retiré · Ctrl+Z » (message), « Retirer de 2 workflows » (menu d'un bloc partagé). « Flux » à côté de "
        "« workflow » : zone « + Flux : Session/Image », raison « Requiert 2 flux filtres, ou empilement couleur ». "
        "Bandeau de l'Inspection : le bloc Session s'appelle « Images » (« Frames » en anglais) alors que le graphe et "
        "la palette disent « Session ». Copies : « Empilement brut (copie) » (Dupliquer) mais « Copie de « Workflow » » "
        "(fenêtre des modèles). Pause : « Pause vérif » (menu, message) mais « Pause définie après Débruitage » "
        "(Journal).",
        shot=str(S / "fr_1366_inspect_gradient1.png"),
        steps=["Chaîne complète en français", "Comparer le menu Exécuter et le bouton de l'Inspection",
               "Clic droit sur un bloc, puis survoler Supprimer dans son panneau ; regarder le bandeau de l'Inspection"])

ex.note("mineur", "Raisons de la palette en style télégraphique, avec des fautes d'élision",
        "Les raisons des blocs grisés ont été raccourcies comme des noms, alors que ce sont des explications : « Pas de "
        "Alignement depuis fichier image », « Pas de Empilement depuis fichier image », « Pas de Astrométrie depuis "
        "fusion » (au lieu de « d'Alignement »…), « Requiert 2 flux filtres, ou empilement couleur », « Poser "
        "Session/Image d'abord », « Requiert demi-empilements (depuis brutes) ». Autres textes : galerie Bi-bande « Des "
        "blocs Session vides sont créés pour couleur : ajoutez-y leurs brutes. » ; infobulle « Balance des blancs de "
        "l'appareil » : « comme le font l'appareil et les dérawtiseurs » (mot inconnu) ; lu dans le catalogue, pas vu "
        "à l'écran : « elle va à Post, ou à le bloc Étoiles/fond, telle quelle ».",
        shot=str(S / "fr_1366_band_hint_log.png"),
        steps=["Projet Image, choisir le bloc Image", "Survoler Alignement, Empilement, Fusion dans la palette"])

ex.note("mineur", "À 1366×768, le bandeau de l'Inspection cache le bloc ouvert",
        "Le bandeau des blocs défile maintenant au lieu d'élargir la fenêtre, mais il ne se place pas sur le bloc ouvert : "
        "en ouvrant Étoiles/fond d'une chaîne complète, son bouton n'est visible que par sa première lettre (« É » ; "
        "« Stars/backgro… » en anglais) au bord droit, et Sortie, Post sont hors de vue derrière la barre de "
        "défilement.",
        shot=str(S / "fr_1366_width_rich_etoiles.png"),
        steps=["Fenêtre 1366×768, chaîne complète avec Gradient et Débruitage placés deux fois",
               "Double-clic sur Étoiles/fond"])

ex.note("mineur", "Ctrl+Z / Ctrl+Y disent seulement « Annulé » / « Rétabli », sans dire quoi",
        "La barre d'état et le Journal écrivent « Annulé » et « Rétabli » sans nommer l'étape. Exemple : désactiver "
        "Gradient 1 (D), puis tenter de supprimer Débruitage (refusé : « Débruitage gardé : requis par Étoiles/fond »), "
        "puis Ctrl+Z : c'est Gradient 1 qui est réactivé, la seule trace est « Annulé ». L'effet est juste (un refus ne "
        "crée pas d'étape), mais l'utilisateur ne voit pas ce qui a été annulé. Même chose en anglais (« Undone » / "
        "« Redone »).",
        shot=str(O / "undo_fr_log.png"),
        steps=["Chaîne complète", "Clic sur Gradient 1, touche D", "Clic sur Débruitage, Suppr (refusé)", "Ctrl+Z"])

ex.note("mineur", "Une image source abîmée ne donne aucun message",
        "Un workflow dont l'image source est un fichier FITS abîmé (illisible) : le panneau du bloc Image montre son nom "
        "et son dossier sans aucun avertissement, l'infobulle du bloc dit « Pas encore calculé », l'Inspection aussi. "
        "Un fichier absent, lui, est signalé (« Fichier introuvable : … »). L'erreur ne viendrait qu'à l'exécution. "
        "Mêmes résultats en anglais.",
        shot=str(O / "errors_fr_damaged_inspect.png"),
        steps=["Projet Image dont le fichier est un FITS abîmé", "Clic sur le bloc Image, puis double-clic"])
print(len(ex.notes()), "notes")
