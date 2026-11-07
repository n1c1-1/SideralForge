"""Engine shapes, round 3 (control): the recheck of the round-2 points and the new anomalies (each seen twice with
explore3_engine_check.py), written to exploration3/engine_shapes.

    python explore3_engine_write.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import OUT_BASE, Explorer  # noqa: E402

OUT = OUT_BASE / "engine_shapes"


def S(name):
    f = OUT / f"{name}.png"
    return str(f) if f.is_file() else None


RECHECK = [
    ("Le même calcul a une autre clé selon le nom interne des blocs : Sortie recalculée en éteignant un bloc, flux "
     "identiques non partagés", "changé",
     "Après un calcul de [Gradient 1, Gradient 2], désactiver Gradient 1 marque maintenant Gradient 2 « Obsolète » "
     "(plus « À jour ») et Exécuter ne refait que l'image de sortie (ligne « Gradient 2 », 1 bloc) ; les clés du moteur "
     "restent inchangées, comme décidé.", "c_offrun_g1_off"),
    ("Une « Pause vérif » sur un bloc posé deux fois (Gradient 2) empêche tout calcul : « Rien à exécuter »", "corrigé",
     "Calcul réel : Pause vérif sur Gradient 2 de [Alignement, Empilement, Gradient 1, Gradient 2] : l'exécution "
     "s'arrête après Gradient 2 (« Pause vérif : Gradient 2 · Inspection, puis Continuer »), Continuer termine.",
     "c_pauserep_g2_paused"),
    ("Image étirée + Étoiles/fond calculés : la page Post reste « Pas encore de calque »", "corrigé",
     "Calcul réel sur un PNG étiré : la page Post montre le calque « Image · Recombiner » ; après « Sans étoiles vers "
     "Post », elle montre aussi « Image · Image sans étoiles ».", "c_stretched_stars_post_after_send"),
    ("Sans Alignement, [Empilement, Gradient] est accepté puis échoue au Gradient avec un message anglais", "changé",
     "Le graphe est toujours accepté et échoue toujours au Gradient pendant le calcul, mais le message est en français "
     "et dit quoi faire (« … placez l'Alignement avant l'Empilement ») ; même chose avec Sélection avant l'Empilement.",
     "c_noreg_stacking_gradient"),
    ("Données impossibles à calculer : Exécuter part quand même, l'échec arrive au milieu, en anglais", "corrigé",
     "Une seule brute, des .fits illisibles, des darks d'une autre taille : Exécuter est grisé avant le calcul avec la "
     "raison en français (et en anglais) ; l'Astrométrie qui échoue dit en français « 19 étoiles seulement … ajoutez des "
     "brutes, ou retirez l'Astrométrie ».", "c_dark_size_refused"),
    ("Dossier des brutes introuvable : Exécuter reste actif (« ≈ 3 h 03 »), le calcul part puis échoue", "corrigé",
     "Exécuter est grisé dès l'ouverture avec « Exécution refusée · … dossier des brutes introuvable : … (son disque "
     "est-il branché ?) » ; seul le message « + ≈ 2 h 21 à exécuter » apparaît encore (voir nouvelle note).",
     "c_missing_folder_refused"),
    ("La méthode d'un bloc posé deux fois (Gradient 2, Débruitage 2) ne peut pas être choisie : grisée « Fixé par les "
     "options du workflow. »", "toujours là",
     "Gradient 1 a ses cartes Méthode et « Modifier échantillons… », Gradient 2 a toujours une liste Méthode grisée sur "
     "Polynôme « Fixé par les options du workflow. » ; même chose pour Débruitage 2.", "c_method_fond2"),
    ("Une branche fait apparaître un bloc « Sortie » dans le graphe et une ligne « Sortie « Image » » au calcul",
     "corrigé",
     "Après une branche sous le Gradient, le graphe n'a que Image, Gradient, Gradient 2 et Post ; la fenêtre "
     "d'exécution liste Image, Gradient, Gradient 2, plus de « Sortie ».", "c_brimg_run_end"),
    ("L'Empilement annonce « drizzle ×1,5 » avant le calcul et « ×1 » après, sans action", "changé",
     "Le bloc dit maintenant « drizzle auto » avant le calcul puis « ×1 » ; mais le panneau montre encore « Échelle du "
     "drizzle 1,50 » avant le calcul, qui devient 1,00 après sans action.", "c_drizzle_panel_before"),
    ("« off » et « on » en anglais dans l'interface française", "corrigé",
     "Les messages disent « Gradient 1 désactivé · Image · Ctrl+Z », « Gradient réactivé · Workflow · Ctrl+Z », le bloc "
     "« désactivé : image transmise », la palette « Alignement déjà là, désactivé : réactiver (D) ».", "c_offon_off"),
    ("Fusion bloquée : F5 dit « Rien à calculer : ajouter brutes ou image » alors qu'un canal a son image", "corrigé",
     "Le bouton dit « Exécuter » sans durée et F5 donne la vraie raison : « Exécution refusée · Fusion : Ha : il faut "
     "l'Astrométrie · OIII : sans fichier image ».", "c_fusion_nofile_f5"),
    ("Pause sur Gradient 1 : la fenêtre d'exécution dit « Pause vérif après Gradient »", "corrigé",
     "Calcul réel : la fenêtre dit « Pause vérif après Gradient 1 », « Pause vérif : Gradient 1 · Inspection, puis "
     "Continuer », la barre d'état « Pause vérif à Gradient 1 ».", "c_pauseg1_paused"),
    ("Sélection posée après Alignement : absente de la fenêtre d'exécution", "corrigé",
     "[Calibration, Alignement, Sélection, Empilement] : la fenêtre liste les quatre blocs, Sélection compris, tous "
     "« Terminé ».", "c_selrow_run_end"),
    ("À l'ouverture d'un projet Image, toute la palette dit « poser Session/Image d'abord »", "corrigé",
     "À l'ouverture, sans bloc choisi, Gradient et Astrométrie sont actifs et un double-clic sur Gradient le pose dans le "
     "flux de l'Image.", "c_cards_opening"),
    ("Le dernier bloc d'un flux porte aussi le menu du workflow", "changé",
     "Le dernier bloc garde les actions du workflow, mais rangées dans un sous-menu « Workflow « … » » (Renommer, "
     "Dupliquer, Enregistrer modèle…, Supprimer workflow) : plus deux « Supprimer » au même niveau.", "c_brimg_graph"),
]

NEW = [
    ("mineur", "Le dernier bloc après une suppression : « Obsolète » au graphe, « À jour » dans le panneau et Inspection",
     "Image [Gradient 1, Gradient 2] calculée, puis Gradient 2 supprimé : le Gradient restant a la barre orange et "
     "l'info-bulle « Obsolète » au graphe, mais le panneau et Inspection disent « À jour » (Inspection dit aussi "
     "« Gradient : non calculé ici ») ; Exécuter propose ≈ 1 min. Rien n'a changé dans le Gradient : seule l'image de "
     "sortie est à refaire. Même chose avec des brutes : [Alignement, Empilement, Gradient] calculé, Gradient supprimé : "
     "Empilement « Obsolète » au graphe, « À jour » dans le panneau, et F5 recalcule une ligne « Empilement ».",
     ["Image → Gradient, Gradient ; Exécuter, attendre la fin", "clic sur Gradient 2, Suppr",
      "survoler le Gradient restant ; cliquer dessus : en-tête du panneau ; onglet Inspection"],
     "c_laststate2_after_delete"),
    ("mineur", "Exécution refusée mais message « + ≈ 2 h 21 à exécuter »",
     "Quand Exécuter est refusé avant le calcul (dossier des brutes introuvable, .fits illisibles, Fusion sans "
     "Astrométrie), le message en bas du graphe annonce quand même du travail : « + ≈ 2 h 21 à exécuter » à "
     "l'ouverture du projet au dossier introuvable, « + ≈ 2 min à exécuter » après avoir posé Alignement et Empilement "
     "sur des .fits illisibles, « + ≈ 4 min à exécuter » à l'ouverture de la Fusion refusée. La barre d'état, elle, dit "
     "« exécution refusée ».",
     ["Session dont le dossier de brutes n'existe plus, Empilement", "ouvrir le projet",
      "lire le message en bas du graphe et la barre d'état"],
     "c_missing_panel"),
    ("mineur", "Le message « … supprimé · Ctrl+Z » ou « … réactivé · Ctrl+Z » remplacé par « + ≈ 1 min à exécuter »",
     "Dans un projet Image, supprimer Gradient 2 (après un calcul) ou réactiver Gradient 1 (touche D) affiche seulement "
     "« + ≈ 1 min à exécuter » : le message qui dit ce qui a été fait et « Ctrl+Z » n'apparaît pas (le Journal a bien la "
     "ligne). Dans un projet de brutes, les mêmes gestes affichent « Gradient supprimé · Ctrl+Z » et « Gradient réactivé "
     "· Workflow · Ctrl+Z ».",
     ["Image → Gradient, Gradient", "clic sur Gradient 1, D, puis D de nouveau ; lire le message",
      "ou : Exécuter, puis clic sur Gradient 2, Suppr ; lire le message"],
     "c_laststate2_after_delete"),
    ("mineur", "Brutes illisibles : la Session dit « 5 / 5 utilisées »",
     "Dossier de 5 .fits qui ne sont pas des FITS : Exécuter est bien refusé (« light_000.fits n'est pas une brute "
     "lisible… : la retirer de la session »), mais le panneau Session dit « Brutes · 5 / 5 utilisées » (« Lights · 5 / 5 "
     "used » en anglais), alors qu'un JPEG ajouté par erreur n'est plus compté.",
     ["Session dont les brutes sont des .fits illisibles", "clic sur la Session : lire « Brutes · … utilisées »",
      "survoler Exécuter"],
     "c_unreadable_refused"),
    ("question", "Astrométrie : le bloc dit « résolue · Gaia DR3 » avant tout calcul",
     "Astrométrie posée (image ou brutes), rien calculé : le bloc dit « Workflow · résolue · Gaia DR3 » avec « Pas encore "
     "calculé » en dessous. Sur l'image de test le calcul échoue ensuite (« 19 étoiles seulement… ») : « résolue » se lit "
     "comme un résultat alors que c'est un réglage.",
     ["Session → Alignement, Empilement, Astrométrie", "lire le bloc Astrométrie"],
     "c_astrometry_summary"),
]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "recheck.jsonl").open("w", encoding="utf-8") as fh:
        for title, status, note, shot in RECHECK:
            fh.write(json.dumps({"title": title, "status": status, "note": note, "shot": S(shot)},
                                ensure_ascii=False) + "\n")
    ex = Explorer("engine_shapes", lang="fr")
    if ex.log.exists():
        ex.log.unlink()
    for sev, title, detail, steps, shot in NEW:
        ex.note(sev, title, detail, shot=S(shot) or False, steps=steps)
    print("recheck", len(RECHECK), "new", len(NEW))
