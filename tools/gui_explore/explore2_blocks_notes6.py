"""block_actions round 2: sixth batch (each reproduced twice)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import Explorer  # noqa: E402

ex = Explorer("block_actions")
D = ex.out
R = "\n(reproduit 2 fois)"
NOTES = [
    ("gênant", "« Supprimer workflow » sur un des deux flux : sa Session reste, sans fil, et se dit « 15 / 15 utilisées »",
     "Projet à deux flux (Session 1 → Calibration → Alignement → Empilement, Session 2 de même). Clic droit sur "
     "l'Empilement du premier flux → « Supprimer workflow » : « Workflow « Workflow » supprimé · Ctrl+Z ». Le bloc "
     "Session 1 reste dessiné, sans aucun fil : aucun workflow ne lit plus ses brutes. Son panneau dit pourtant « Brutes · "
     "15 / 15 utilisées ». Rien ne dit que cette session n'est plus traitée. Ctrl+Z rend le flux.",
     ["Projet à deux flux, chaîne dans chaque", "Clic droit Empilement du premier flux → Supprimer workflow",
      "Cliquer Session 1"],
     D / "dig_two_flows_wf_delete_1.png"),
    ("question", "Un clic dans le vide du graphe ne désélectionne pas : Suppr et D visent encore le bloc choisi",
     "Chaîne jusqu'à Couleur. Suppr sur Gradient puis Ctrl+Z : Gradient revient, mais c'est Astrométrie qui reste "
     "choisie (pas le bloc rendu). Un clic dans une zone vide du graphe laisse Astrométrie choisie (cadre bleu, panneau) ; "
     "Suppr et D s'appliquent alors à elle (ici refusés avec la raison). Dans les éditeurs de référence, un clic dans le "
     "vide désélectionne, et Annuler resélectionne l'objet rendu.",
     ["Chaîne jusqu'à Couleur", "Cliquer Gradient, Suppr, Ctrl+Z", "Clic dans le vide du graphe, Suppr"],
     D / "dig_empty_click_1.png"),
]
for sev, title, detail, steps, shot in NOTES:
    ex.note(sev, title, detail + R, shot=str(shot), steps=steps)
print(len(NOTES), "noted")
