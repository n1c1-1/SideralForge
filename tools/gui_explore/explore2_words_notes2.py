"""Words and layout, round 2, second batch of notes (appends to <out>/anomalies.jsonl; run once). Reproduced by
explore2_words_height.py fr / en and by explore2_words_walk.py fr 1366 768 rich (twice)."""
from __future__ import annotations

from harness import Explorer

ex = Explorer("words_layout", lang="fr", size=(1366, 768))
ex.note("gênant", "À 1366×768, l'Inspection d'une session agrandit la fenêtre en hauteur au-delà de l'écran, et elle le reste",
        "Double-clic sur une Session : la liste « Vérifications » n'a pas de barre de défilement, elle impose sa hauteur "
        "à toute la fenêtre. Avec deux sessions et 16 avertissements la fenêtre passe de 768 à 920 px de haut, à 1120 px "
        "avec le Journal ouvert ; le bas (ligne d'état, Journal, bouton Journal) sort d'un écran 768. De retour sur le "
        "Graphe la fenêtre garde cette hauteur et refuse d'être ramenée à 768 (hauteur minimale 920). La hauteur suit "
        "le nombre d'avertissements : un dossier ajouté par erreur avec beaucoup de fichiers illisibles l'agrandit "
        "d'autant. Même chose en anglais.",
        shot=str(ex.out / "height_fr_inspect_session_log.png"),
        steps=["Fenêtre 1366×768, projet à deux sessions", "Double-clic sur Session 1",
               "Affichage > Journal ; revenir au Graphe ; essayer de ramener la fenêtre à 768 px"])
print(len(ex.notes()), "notes")
