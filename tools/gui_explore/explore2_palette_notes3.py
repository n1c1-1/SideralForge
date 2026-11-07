"""palette_matrix round 2: third batch of anomalies (each reproduced twice)."""
from __future__ import annotations

from explore2_palette_lib import Matrix

m = Matrix("notes3")
ex = m.ex
D = ex.out
notes = [
    ("bloquant", "Plusieurs nuits : ajouter une Session détache les nuits déjà là du workflow",
     "Projet fait avec le modèle « Plusieurs nuits » (Session 1 et Session 2, 8 brutes chacune, une chaîne complète). "
     "Double-clic sur la carte Session, ou carte Session lâchée sur « + Session » : la Session 3 (vide) est créée, mais le "
     "workflow « Plusieurs nuits » ne prend plus que la Session 3 ; Session 1 et Session 2 restent dans la colonne des "
     "sources sans aucun lien (sessions du workflow : [1, 2] → [3]). Exécuter reste actif (« ≈ 9 min ») alors que la "
     "seule session du workflow n'a aucune brute. Ctrl+Z remet bien l'état d'avant. Même chose dans un projet à deux "
     "sessions et un workflow vide. Dans un projet à deux flux (une session chacun), la nouvelle session s'ajoute bien "
     "au flux choisi. Attendu (PRINCIPLES : un bloc se pose et se relie seul ; rien ne tourne sans données) : la Session "
     "3 rejoint les deux autres.", str(D / "nuits_dbl_apres.png"),
     ["nouveau projet, modèle « Plusieurs nuits », des brutes dans chaque session", "double-clic carte Session"]),
    ("mineur", "Message « pourquoi grisé » resté affiché après Ctrl+Y, sur une carte devenue disponible",
     "Workflow de brutes : Calibration, Alignement, Empilement posés, puis Ctrl+Z (Empilement retiré). La souris sur la "
     "carte Astrométrie grisée : le message « Astrométrie requiert Empilement avant » s'affiche. Ctrl+Y (souris "
     "immobile) : Empilement revient, la carte Astrométrie redevient disponible, mais le message reste affiché (il "
     "recouvre le bloc Session) tant que la souris ne quitte pas la carte. Attendu : le message disparaît quand la carte "
     "n'est plus grisée.", str(D / "hint_apres_ctrl_y.png"),
     ["projet Session (workflow vide)", "double-clic Calibration, Alignement, Empilement", "Ctrl+Z",
      "souris sur la carte Astrométrie", "Ctrl+Y"]),
    ("mineur", "Après Ctrl+Z d'un bloc posé, aucun bloc choisi et panneau de droite vide",
     "Workflow de brutes : Calibration, Alignement, Empilement posés par double-clic (Empilement choisi, son panneau à "
     "droite). Ctrl+Z : Empilement retiré, mais aucun bloc n'est surligné et le panneau de droite est vide ; le bloc "
     "choisi est « Sélection », qui n'est pas posé (Suppr, D et Entrée ne font alors rien). Même chose après Ctrl+Y. "
     "Dans un workflow Image, Ctrl+Z choisit bien le bloc d'avant (Astrométrie). Attendu : le bloc précédent choisi, "
     "son panneau à droite.", str(D / "sel_apres_ctrl_y_session_empty_wf.png"),
     ["projet Session (workflow vide)", "double-clic Calibration, Alignement, Empilement", "Ctrl+Z"]),
]
for sev, title, detail, shot, steps in notes:
    ex.note(sev, title, detail, shot=shot, steps=steps)
ex.close()
