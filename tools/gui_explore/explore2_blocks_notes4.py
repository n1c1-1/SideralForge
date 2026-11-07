"""block_actions round 2: fourth batch (each reproduced twice)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import Explorer  # noqa: E402

ex = Explorer("block_actions")
D = ex.out
R = "\n(reproduit 2 fois)"
NOTES = [
    ("bloquant", "Poser une deuxième Session depuis la palette débranche la première : ses brutes ne sont plus lues",
     "Projet à une session (Session 1, 15 brutes) avec Calibration → Alignement → Empilement. Double-clic sur la carte "
     "Session de la palette, ou Session glissée sur « + Session » : une Session 2 vide (« brutes à ajouter ») est posée "
     "et le workflow ne lit plus qu'elle. Session 1 n'a plus de fil vers Calibration, Post passe à « 0 calque », "
     "l'estimation tombe à ≈ 1 min. Aucun message. Ctrl+Z rend l'état d'avant. Par « Ajouter session… » (menu de la "
     "Session), au contraire, les deux sessions restent lues.",
     ["Projet à une session + Calibration, Alignement, Empilement", "Double-clic sur la carte Session de la palette"],
     D / "dig_join_dbl_1.png"),
]
for sev, title, detail, steps, shot in NOTES:
    ex.note(sev, title, detail + R, shot=str(shot), steps=steps)
print(len(NOTES), "noted")
