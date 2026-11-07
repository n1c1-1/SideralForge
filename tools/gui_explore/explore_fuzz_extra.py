"""Fuzz area, one anomaly seen while fuzzing that no invariant checks: the "Already stretched image" template.

    python explore_fuzz_extra.py
"""
import os
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_fuzz as F  # noqa: E402

fz = F.Fuzz("fr")
for k in range(2):                                    # twice, fresh projects
    fz.start({"template": "image_stretched", "lang": "fr"}, f"x{k}")
    s = fz.ex.s
    got = [(q.name, (q.source or {}).get("stretched"), s.display_keys(q)) for q in s.paths]
    print(k, got, flush=True)
ok = got and got[0][1] is False and "astrometrie" in got[0][2]
if ok:
    fz.ex.note("gênant", "Le modèle « Image déjà étirée » crée une chaîne d'image linéaire",
               "Nouveau projet, fenêtre des modèles, onglet « À partir d'une image », « Image déjà étirée » (une image PNG choisie), Créer : "
               "le workflow obtenu a une Image de type « linéaire » suivie d'Astrométrie, Gradient et Couleur, au lieu "
               "d'une image étirée qui va au Post telle quelle (le modèle porte bien « étirée », mais le type de source "
               "« image » l'emporte). Reproduit 2/2 ; toutes les séquences du fuzz parties de ce modèle l'ont.",
               shot=True, steps=["Nouveau projet", "fenêtre des modèles : onglet « À partir d'une image » → « Image déjà étirée »",
                                 "choisir un fichier PNG", "Créer"])
os._exit(0)
