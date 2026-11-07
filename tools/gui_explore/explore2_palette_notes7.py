"""palette_matrix round 2: seventh batch (each reproduced twice)."""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("notes7")
ex = m.ex
D = ex.out
notes = [
    ("mineur", "Messages après une pose : « + ≈ ≤ 1 min à exécuter », et la branche nommée par son point de départ",
     "Image déjà étirée, double-clic sur Étoiles/fond : le message dit « + ≈ ≤ 1 min à exécuter » (deux signes "
     "d'approximation de suite). Workflow complet, carte Gradient lâchée sur la place « nouvelle branche » sous Empilement : le "
     "message dit « Nouvelle branche · Empilement · Ctrl+Z » (on lit qu'un Empilement a été posé), alors que la "
     "branche créée s'appelle « Full · Gradient » et ne contient de nouveau qu'un Gradient. Attendu (PRINCIPLES : même nom partout ; "
     "mots simples) : « + ≤ 1 min à exécuter » ; « Nouvelle branche · Gradient · Ctrl+Z ».",
     str(D / "etiree_etoiles_recombiner.png"),
     ["nouveau projet, modèle « Image déjà étirée »", "clic sur l'Image, double-clic Étoiles/fond",
      "projet workflow complet, glisser Gradient sur « nouvelle branche » sous Empilement"]),
]
for sev, title, detail, shot, steps in notes:
    ex.note(sev, title, detail, shot=shot, steps=steps)
ex.close()
