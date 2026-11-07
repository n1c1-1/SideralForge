"""palette_matrix round 2: eighth batch (each reproduced twice)."""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("notes8")
ex = m.ex
D = ex.out
notes = [
    ("mineur", "Le mot anglais « off » dans l'interface française",
     "Workflow complet, Gradient choisi, touche D : le message dit « Gradient off · Full · Ctrl+Z ». Alignement "
     "désactivé de même : la carte Alignement de la palette est grisée avec « Alignement déjà là, off : réactiver "
     "(D) ». Attendu (PRINCIPLES : en français, aucun mot anglais sauf « workflow ») : « Gradient désactivé · Full · "
     "Ctrl+Z », « Alignement déjà là, désactivé : réactiver (D) ».", str(D / "off_registration.png"),
     ["projet workflow complet", "clic sur Alignement, touche D", "survoler la carte Alignement"]),
]
for sev, title, detail, shot, steps in notes:
    ex.note(sev, title, detail, shot=shot, steps=steps)
ex.close()
