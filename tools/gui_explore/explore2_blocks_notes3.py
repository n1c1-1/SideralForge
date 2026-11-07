"""block_actions round 2: third batch of anomalies (each reproduced twice)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import Explorer  # noqa: E402

ex = Explorer("block_actions")
D = ex.out
R = "\n(reproduit 2 fois)"
NOTES = [
    ("gênant", "La Session vide d'un canal de Fusion ne se retire jamais : elle revient aussitôt, le message dit « retirée »",
     "Projet mono Ha + OIII avec Fusion. Croix de la Session Ha : une Session « Ha » vide (« brutes à ajouter ») la "
     "remplace. Sa croix est dessinée ; un clic dessus affiche « « Ha » retirée · fichiers gardés · Ctrl+Z », et une "
     "nouvelle Session « Ha » vide reparaît à la même place (une session de plus est créée à chaque fois). Même chose "
     "avec « Supprimer » de son menu. Le bloc ne part jamais, et chaque essai ajoute une étape à annuler. Si elle ne peut "
     "pas partir (la Fusion a besoin d'un canal Ha), il faudrait le refuser avec la raison.",
     ["Projet mono, Astrométrie dans chaque, Fusion", "Croix de la Session Ha", "Croix de la Session « Ha » vide",
      "Clic droit → Supprimer"],
     D / "dig_mono_placeholder_1.png"),
    ("mineur", "Désactiver un bloc retire sa pause sans le dire, et la réactiver ne la rend pas",
     "Chaîne jusqu'à Couleur. Clic droit Gradient → « Pause vérif » (cochée dans le menu). Puis D sur Gradient : "
     "message « Gradient off · Workflow · Ctrl+Z » ; la pause a disparu. D de nouveau : Gradient revient, la pause non. "
     "Deux Ctrl+Z la rendent. Même chose sur l'Astrométrie d'un canal de Fusion.",
     ["Chaîne jusqu'à Couleur", "Clic droit Gradient → Pause vérif", "Cliquer Gradient, D, puis D"],
     D / "dig_pause_off_1.png"),
    ("mineur", "« Vers Post » sur la Fusion : aucun message, deux calques dont un au nom anglais",
     "Projet mono Ha + OIII avec Fusion. Clic droit Fusion → « Vers Post » : rien ne s'affiche (pas de « … vers Post · "
     "Ctrl+Z » comme pour les autres blocs). La composition reçoit deux calques, « HOO » et « HOO · Recombine » (mot "
     "anglais). Ctrl+Z et Ctrl+Y marchent.",
     ["Projet mono, Astrométrie dans chaque, Fusion", "Clic droit Fusion → Vers Post"],
     None),
]
for sev, title, detail, steps, shot in NOTES:
    ex.note(sev, title, detail + R, shot=str(shot) if shot else False, steps=steps)
print(len(NOTES), "noted")
