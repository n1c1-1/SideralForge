"""block_actions round 2: fifth batch (each reproduced twice)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import Explorer  # noqa: E402

ex = Explorer("block_actions")
D = ex.out
R = "\n(reproduit 2 fois)"
NOTES = [
    ("question", "Changer le filtre de la session d'un canal de Fusion : la Fusion ne dit rien",
     "Projet mono Ha + OIII avec Fusion HOO. Cliquer la Session Ha, Filtre → « SII » (ou « OIII », ou « aucun ») : "
     "accepté sans message ; la Fusion reste « HOO · Ha, OIII », sans cadre rouge ni avertissement, Exécuter reste "
     "actif. Le canal Ha de la Fusion lit donc des brutes marquées SII (ou deux canaux OIII). Ctrl+Z rend le filtre. "
     "Faut-il avertir, ou refuser avec la raison ?",
     ["Projet mono, Astrométrie dans chaque, Fusion", "Cliquer Session Ha, Filtre → SII"],
     D / "dig_filter_fusion_SII.png"),
    ("mineur", "Le panneau d'une Session n'a pas de bouton « Supprimer »",
     "Projet à deux sessions (ou mono) : le menu du clic droit d'une Session a « Supprimer » et sa croix marche, mais "
     "son panneau n'a que le nom, le filtre, les brutes par type et « Revoir images… » : pas de « Supprimer », alors que "
     "tous les autres blocs l'ont en tête de panneau. (Le × de la ligne du dossier retire les fichiers, pas la session.)",
     ["Projet à deux sessions", "Cliquer Session 1"],
     D / "dig_filter_fusion_SII.png"),
]
for sev, title, detail, steps, shot in NOTES:
    ex.note(sev, title, detail + R, shot=str(shot), steps=steps)
print(len(NOTES), "noted")
