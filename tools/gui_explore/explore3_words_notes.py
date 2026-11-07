"""Words and layout, round 3: the new problems, each reproduced twice (two languages or two sizes) by
explore3_words_points.py, explore3_words_runoff.py and explore3_words_errors.py. Appends to <out>/anomalies.jsonl
(run once)."""
from __future__ import annotations

from harness import Explorer

ex = Explorer("words_layout", lang="fr", size=(1366, 768))
P = ex.out / "points"
ex.note("gênant", "À 1366×768, ouvrir un bloc de droite dans l'Inspection fait sortir « ◂ Graphe » de l'écran",
        "Le bandeau des blocs défile maintenant jusqu'au bloc ouvert, mais il emporte avec lui le bouton « ◂ Graphe » et "
        "la liste des workflows : en ouvrant Étoiles/fond d'une chaîne complète, « ◂ Graphe » n'est plus visible du "
        "tout et la liste ne montre que « ow » (« orkflow » en anglais). Pour revenir au graphe il faut défiler le "
        "bandeau vers la gauche, ou savoir qu'Échap y ramène. À 1920×1080, tout reste visible. Vu en français et en "
        "anglais.",
        shot=str(P / "fr_1366_inspect_stars_band.png"),
        steps=["Fenêtre 1366×768, projet à une session", "Chaîne complète avec Gradient et Débruitage placés deux fois",
               "Double-clic sur Étoiles/fond", "Regarder le début du bandeau des blocs"])
ex.note("gênant", "« Jusqu'ici » sur un bloc désactivé : « Gradient 2 n'est plus dans le graphe », alors qu'il y est",
        "Gradient 2 désactivé (menu Désactiver), puis choisi et « Jusqu'ici » : la fenêtre d'exécution s'intitule "
        "« Exécuter « Workflow » jusqu'à Gradient 2 », dit en rouge « Gradient 2 n'est plus dans le graphe » et "
        "grise Exécuter. Le bloc est pourtant dessiné dans le graphe, seulement désactivé : la raison est fausse, elle "
        "devrait dire qu'il est désactivé et comment le réactiver (D). Même chose en anglais à 1920×1080 "
        "(« Gradient 2 is no longer in the graph »).",
        shot=str(ex.out / "runoff_fr_1366.png"),
        steps=["Projet à une session, chaîne jusqu'à Couleur avec Gradient placé deux fois",
               "Clic droit sur Gradient 2 > Désactiver", "Clic sur Gradient 2", "Exécuter > Jusqu'ici"])
ex.note("mineur", "Infobulle d'une image source abîmée : deux points à la suite (« abîmé.. »)",
        "Le fichier FITS abîmé est maintenant signalé, mais l'infobulle du bloc Image dit « Ne peut pas s'exécuter : "
        "bad_damaged.fits ne peut pas être lu : le fichier est abîmé.. L'exécution est refusée tant que ce n'est pas "
        "réglé. » (en anglais « the file is damaged.. Run is refused »). Pour un fichier absent, un seul point.",
        shot=str(ex.out / "errors_fr_damaged_inspect.png"),
        steps=["Projet Image dont le fichier est un FITS abîmé", "Survoler le bloc Image dans le graphe"])
print(len(ex.notes()), "notes")
