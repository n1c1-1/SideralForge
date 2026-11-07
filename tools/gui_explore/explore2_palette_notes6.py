"""palette_matrix round 2: sixth batch of anomalies (each reproduced twice)."""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("notes6")
ex = m.ex
D = ex.out
notes = [
    ("gênant", "Session ajoutée dans un projet de filtres : elle rejoint un canal sans son filtre, parfois pas celui "
     "affiché",
     "Projet du modèle SHO, la Fusion choisie (palette « pour « SHO (palette Hubble) » »). Double-clic sur la carte "
     "Session : la Session 4 est créée sans filtre (« aucun (capteur couleur) ») et ajoutée au workflow SII (sessions "
     "[1] → [1, 4]), alors que l'en-tête de la palette nommait la Fusion ; rien ne le dit. Projet mono (Ha, OIII), Ha "
     "choisi : la Session 3 rejoint Ha (logique) mais sans filtre, c'est-à-dire une session couleur dans un workflow "
     "Ha. Attendu (PRINCIPLES : un bloc se pose et se relie seul ; la portée d'une action est visible ; le filtre de la "
     "Session) : la nouvelle session prend le filtre du canal qu'elle rejoint, et rejoint le workflow nommé dans "
     "l'en-tête (ou le dit).", str(D / "session_filtre_sho.png"),
     ["nouveau projet, modèle SHO", "double-clic carte Session", "regarder le filtre de la Session 4 et le workflow SII"]),
]
for sev, title, detail, shot, steps in notes:
    ex.note(sev, title, detail, shot=shot, steps=steps)
ex.close()
