"""block_actions round 2: anomalies found after the first batch (each reproduced twice)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import Explorer  # noqa: E402

ex = Explorer("block_actions")
D = ex.out
R = "\n(reproduit 2 fois)"
NOTES = [
    ("gênant", "« Essayer en variante » : la variante reçoit un bloc Sélection que l'original n'a pas, et « Garder » l'ajoute",
     "Chaîne Calibration → Alignement → Empilement → Astrométrie → Gradient → Couleur (sans Sélection). Le projet "
     "est mis dans l'état « résultats calculés » (rien n'est calculé ici : seul le choix Remplacer / Essayer en variante "
     "est provoqué). Changer « Seuil des pixels chauds » de Calibration (0,6 → 0,9) → « Essayer en variante » : la "
     "variante « Workflow · seuil des pixels chauds 0,9 » est dessinée avec Calibration → Alignement → Sélection → "
     "Empilement → … → Couleur → Sortie : un bloc Sélection que personne n'a posé, et un bloc « Sortie ». Le coût passe "
     "de ≈ 8 min à ≈ 16 min. « Garder » remplace alors le workflow par la variante : « Workflow » contient désormais "
     "Sélection. « Garder », « Abandonner » et « Remplacer » n'affichent aucun message. Ctrl+Z et Ctrl+Y marchent.",
     ["Chaîne sans Sélection", "Calibration : Seuil des pixels chauds 0,9, Entrée", "Essayer en variante",
      "Cliquer la variante → Garder"],
     D / "variant_made_fr_1.png"),
]
for sev, title, detail, steps, shot in NOTES:
    ex.note(sev, title, detail + R, shot=str(shot), steps=steps)
print(len(NOTES), "noted")
