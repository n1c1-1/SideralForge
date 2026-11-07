"""palette_matrix round 2: anomalies found after the first batch of notes (each reproduced twice)."""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("notes2")
ex = m.ex
D = ex.out
notes = [
    ("mineur", "Branche avec un second Gradient : « Gradient » et « Gradient 2 », jamais « Gradient 1 »",
     "Workflow complet. Carte Gradient lâchée sur la place « nouvelle branche » juste après Gradient : une branche "
     "« Full · Gradient 2 » est créée (Gradient partagé, puis un second Gradient). Dans le graphe, le panneau et la "
     "fenêtre « Exécuter jusqu'ici », la branche contient « Gradient » puis « Gradient 2 ». Dans le tronc, deux Gradient "
     "posés l'un après l'autre s'appellent bien « Gradient 1 » et « Gradient 2 ». Attendu (PRINCIPLES : les copies sont "
     "numérotées de gauche à droite, même nom partout) : « Gradient 1 » et « Gradient 2 » dans la branche aussi.",
     str(D / "branche_gradient_apres_gradient.png"),
     ["projet workflow complet", "glisser la carte Gradient sur « nouvelle branche » après Gradient"]),
    ("mineur", "1366×768 : en glissant un bloc, la plupart des places sont hors de l'écran",
     "Workflow complet, fenêtre 1366×768. Carte Gradient tenue au-dessus du graphe : seules les trois premières places "
     "(après Empilement, après Astrométrie, après Gradient) sont visibles ; les suivantes, et Post, sont sous le panneau "
     "de droite, et le texte des blocs fait quelques pixels. À l'ouverture déjà, Recombiner et Post ne sont pas à "
     "l'écran. Attendu (PRINCIPLES : lisible en 1366×768 ; pendant le glisser les places valides s'allument) : toutes "
     "les places visibles ou atteignables pendant le glisser.",
     str(D / "layout_fr_1366x768_legacy_drag.png"),
     ["fenêtre 1366×768", "projet workflow complet", "glisser la carte Gradient sans lâcher"]),
]
for sev, title, detail, shot, steps in notes:
    ex.note(sev, title, detail, shot=shot, steps=steps)
ex.close()
