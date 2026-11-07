"""Words and layout, round 2, third batch (appends to <out>/anomalies.jsonl; run once). Reproduced by
explore2_words_height.py fr and en (two sessions, Inspect)."""
from __future__ import annotations

from harness import Explorer

ex = Explorer("words_layout", lang="fr", size=(1366, 768))
ex.note("mineur", "Inspection, projet à deux sessions : le bandeau des blocs n'a plus de Calibration",
        "Avec deux sessions, le graphe montre un bloc Calibration par session, mais le bandeau de l'Inspection passe "
        "de « Images » à « Alignement » (« Frames › Registration » en anglais) : aucune des deux Calibration n'y est, "
        "on ne peut les ouvrir que depuis le graphe. Avec une session, le bandeau a bien « Calibration ».",
        shot=str(ex.out / "height_en_inspect_fond_log.png"),
        steps=["Projet à deux sessions, chaîne complète", "Double-clic sur Gradient", "Lire le bandeau des blocs"])
print(len(ex.notes()), "notes")
