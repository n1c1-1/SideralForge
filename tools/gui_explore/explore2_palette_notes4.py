"""palette_matrix round 2: fourth batch of anomalies (each reproduced twice)."""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("notes4")
ex = m.ex
D = ex.out
notes = [
    ("gênant", "Fusion retirée : les blocs qu'elle avait ajoutés aux canaux restent, et deux Recombiner apparaissent",
     "Projet mono (Ha, OIII : Calibration → Alignement → Empilement, deux calques dans Post). Double-clic Fusion : "
     "Débruitage et Étoiles/fond sont ajoutés à chaque canal. Puis la Fusion glissée sur la palette (ou Supprimer) : "
     "message « Fusion et suivants retirés · Ctrl+Z ». Vu : la Fusion part, mais Débruitage et Étoiles/fond restent "
     "dans Ha et OIII, et chaque canal a maintenant un bloc Recombiner vers Post. Le projet n'est pas revenu à ce qu'il "
     "était avant la Fusion. (Ctrl+Z, lui, remet exactement.) Attendu : retirer la Fusion retire aussi ce que sa pose "
     "avait ajouté, ou le message dit ce qui reste.", str(D / "dragout_mono_fusion.png"),
     ["projet mono Ha + OIII", "double-clic carte Fusion", "glisser le bloc Fusion sur la palette"]),
    ("mineur", "Messages de retrait : accords fautifs et phrases tronquées",
     "Bloc glissé sur la palette : « Calibration retiré · Ctrl+Z » (retirée), « Astrométrie gardé : requis par Couleur » "
     "(gardée), « Empilement gardé : suite requiert image » (ne dit pas clairement que les blocs suivants ont besoin de "
     "son image). Attendu (PRINCIPLES : mots simples, traduction française complète) : un français correct qui dit "
     "pourquoi.", str(D / "dragout_legacy_astrometrie.png"),
     ["projet workflow complet", "glisser le bloc Astrométrie sur la palette", "glisser le bloc Calibration sur la palette"]),
]
for sev, title, detail, shot, steps in notes:
    ex.note(sev, title, detail, shot=shot, steps=steps)
ex.close()
