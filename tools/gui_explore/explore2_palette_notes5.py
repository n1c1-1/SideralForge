"""palette_matrix round 2: fifth batch of anomalies (each reproduced twice)."""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("notes5")
ex = m.ex
D = ex.out
notes = [
    ("gênant", "Modèle SHO, avant d'ajouter les brutes : les trois filtres dessinés comme une seule chaîne, et trois "
     "blocs « Sortie »",
     "Nouveau projet, modèle « SHO (palette Hubble) » (sessions SII, Ha, OIII encore vides). Le graphe montre trois "
     "Calibrations qui se rejoignent dans un seul Alignement, puis une seule chaîne Sélection → Empilement → Astrométrie "
     "→ Gradient → Débruitage → Étoiles/fond pour les trois workflows (comme si les trois filtres étaient empilés "
     "ensemble), suivie de trois blocs « Sortie » (« SII · workflow », « Ha · workflow », « OIII · workflow ») et d'une "
     "Fusion bordée de rouge. Dès que chaque session a ses brutes, le graphe montre trois chaînes séparées et plus de "
     "blocs Sortie. Attendu (PRINCIPLES : la Sortie n'est pas un bloc à l'écran ; un graphe qui montre ce qui sera "
     "calculé) : trois chaînes dès le départ, sans bloc Sortie.", str(D / "sho_modele_ouverture.png"),
     ["nouveau projet, modèle « SHO (palette Hubble) »", "regarder le graphe avant d'ajouter des brutes"]),
]
for sev, title, detail, shot, steps in notes:
    ex.note(sev, title, detail, shot=shot, steps=steps)
ex.close()
