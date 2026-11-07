"""Engine shapes, round 2: the anomalies found (each reproduced at least twice by the explore2_engine_* scripts),
written to exploration2/engine_shapes/anomalies.jsonl with the screenshot of each.

    python explore2_engine_notes.py
"""
import json
import time
from pathlib import Path

OUT = Path("F:/astro_essais/A_REGARDER/exploration2/engine_shapes")


def S(name):
    f = OUT / f"{name}.png"
    return str(f) if f.is_file() else ""


NOTES = [
    ("bloquant", "Une « Pause vérif » sur un bloc posé deux fois (Gradient 2) empêche tout calcul : « Rien à exécuter »",
     "Graphe [Alignement, Empilement, Gradient 1, Gradient 2] ou image [Gradient 1, 2, 3], rien encore calculé. Pause vérif "
     "sur Gradient 2 (le menu l'accepte, message « Pause vérif après Gradient 2 »). Exécuter (bouton ou F5) ouvre la "
     "fenêtre « Exécuter projet » vide : « Rien à exécuter », aucune ligne ; rien ne se calcule, ni avant ni après la pause. "
     "La même pause sur Gradient 1 marche (calcul jusqu'à Gradient 1, puis Continuer).",
     ["Image ou session ; poser Gradient deux fois", "clic droit sur Gradient 2 → Pause vérif", "Exécuter"],
     "pauserep2_g2"),
    ("bloquant", "Image étirée + Étoiles/fond calculés : la page Post reste « Pas encore de calque »",
     "Image PNG marquée étirée, bloc Étoiles/fond (le Recombiner se pose avec), calcul réel terminé (2 blocs). Le bloc "
     "Post du graphe dit « 1 calque », mais la page Post affiche « Pas encore de calque · Chaque workflow calculé donne ici "
     "un calque ». Après « Sans étoiles vers Post » (message « Image sans étoiles · « Image » vers Post »), le bloc dit "
     "« 2 calques » et la page reste vide. Le même graphe à partir de brutes (Alignement… Débruitage, Étoiles/fond) donne "
     "bien son calque « Workflow · Recombiner ». Le calque prévu par le graphe s'appelle « Image · Recombine » (anglais).",
     ["Image étirée (PNG)", "poser Étoiles/fond", "Exécuter, attendre la fin", "onglet Post",
      "Graphe : clic droit Étoiles/fond → Sans étoiles vers Post ; onglet Post"],
     "edge_stretched_split2_post_after_send"),
    ("gênant", "Sans Alignement, [Empilement, Gradient] est accepté puis échoue au Gradient avec un message anglais",
     "Session de brutes Bayer qui bougent de quelques pixels. Graphes [Empilement, Gradient], [Calibration, Empilement, "
     "Gradient] et [Sélection, Empilement, Gradient] : la palette accepte tout, Exécuter part, l'Empilement se calcule, "
     "puis « Gradient en échec : too few covered areas to fit a background ». Avec Alignement, le même graphe passe. Soit "
     "le graphe est refusé avant (avec la raison en français), soit il doit tourner.",
     ["Session de brutes", "poser Empilement puis Gradient (sans Alignement)", "Exécuter"],
     "sweep_stacking_gradient_failed"),
    ("gênant", "Données impossibles à calculer : Exécuter part quand même, l'échec arrive au milieu, en anglais",
     "Trois cas, chacun lancé sans avertissement avant le calcul : (1) une seule brute : Empilement en échec « fewer than "
     "2 usable frames after selection; relax the selection thresholds » (il n'y a pas de bloc Sélection) ; (2) des .fits "
     "qui ne sont pas des FITS : la Session dit « 5 / 5 utilisées », puis la ligne Alignement affiche l'erreur Python brute "
     "« OSError: No SIMPLE card found, this file does not appear to be a valid FITS file… » ; (3) des darks d'une autre "
     "taille que les brutes : Calibration en échec « 3 darks frame(s) have a different size than the lights, e.g. "
     "<chemin> ». Le dossier introuvable, lui, a un message français. Un échec normal en cours de calcul reste aussi en "
     "anglais : Astrométrie sur un champ sans assez d'étoiles « too few stars detected in the stack to solve the field ».",
     ["Session d'1 brute → Alignement, Empilement → Exécuter",
      "Session de faux .fits → Alignement, Empilement → Exécuter",
      "Session avec darks 100×150 et brutes 200×300 → Calibration, Alignement, Empilement → Exécuter"],
     "edge_unreadable_failed"),
    ("gênant", "Dossier des brutes introuvable : Exécuter reste actif (« ≈ 3 h 03 »), le calcul part puis échoue",
     "Session dont le dossier n'existe plus (disque débranché). Le panneau dit « 📁 nowhere · 0 fichier » et « Brutes · "
     "0 / 0 utilisées » sans avertissement ; le bouton dit « Exécuter · ≈ 3 h 03 ». Le calcul part et échoue à "
     "l'Empilement : « dossier des brutes introuvable : … (son disque est-il branché ?) ». Le message est bon, mais il "
     "devrait venir avant le calcul (« rien ne tourne sans données »).",
     ["Session → son dossier de brutes supprimé ou débranché", "Empilement", "Exécuter"],
     "recheck_missing_folder"),
    ("gênant", "La méthode d'un bloc posé deux fois (Gradient 2, Débruitage 2) ne peut pas être choisie : grisée « Fixé "
     "par les options du workflow. »",
     "Le premier Gradient a ses cartes Méthode (Polynôme, RBF, Automatique) et « Modifier échantillons… ». Un second "
     "Gradient posé depuis la palette a une liste « Méthode » grisée sur Polynôme, avec l'infobulle « Fixé par les options "
     "du workflow. », et pas de « Modifier échantillons… ». On ne peut donc pas faire « Polynôme puis RBF » depuis "
     "l'interface. Même chose pour Débruitage 2 (liste grisée sur « Réseau pré-entraîné », alors que Débruitage 1 a ses "
     "deux cartes). Vu dans des flux Image et de brutes.",
     ["Image → poser Gradient deux fois", "cliquer Gradient 1 : cartes Méthode", "cliquer Gradient 2 : liste grisée",
      "Session → Alignement, Empilement, Débruitage, Gradient, Débruitage : cliquer Débruitage 2"],
     "method_fond2"),
    ("gênant", "Une branche fait apparaître un bloc « Sortie » dans le graphe et une ligne « Sortie « Image » » au calcul",
     "Une branche posée sous le dernier bloc d'un flux (Gradient) : le flux d'origine se termine par un bloc grisé "
     "« Sortie · Image · vers Post », relié au Post. Son menu : Renommer, Dupliquer, Enregistrer comme modèle…, Supprimer "
     "workflow, Supprimer, « Désactiver — absent du workflow ». La fenêtre d'exécution a une ligne « Sortie « Image » » "
     "(« Sortie « Workflow » » en brutes). Règle : la Sortie n'est pas un bloc à l'écran.",
     ["Image → Gradient", "glisser Gradient sur la place « Une nouvelle branche d'ici » sous le Gradient",
      "regarder le graphe, clic droit sur Sortie, Exécuter"],
     "brimg_graph"),
    ("question", "L'Empilement annonce « drizzle ×1,5 » avant le calcul et « ×1 » après, sans action",
     "Avant le premier calcul le bloc dit « drizzle ×1,5 · rejet à 5 σ » ; une fois calculé il dit « drizzle ×1 · rejet à "
     "5 σ » et reste « À jour ». La valeur suggérée change avec les données lues. Le bloc annonce donc un réglage qui ne sera "
     "pas utilisé. Voulu ? (vu sur 5 projets de brutes 200×300)",
     ["Session → Alignement, Empilement", "lire le bloc Empilement", "Exécuter, relire le bloc"],
     "outd_after_run"),
    ("mineur", "« off » et « on » en anglais dans l'interface française",
     "Désactiver un bloc : message « Alignement off · Workflow · Ctrl+Z », résumé du bloc « off : image transmise », raison "
     "de la palette « Alignement déjà là, off : réactiver (D) » ; réactiver : « Gradient 1 on · Image · Ctrl+Z ».",
     ["poser un bloc, touche D", "survoler sa carte dans la palette", "touche D de nouveau"],
     "offrun_g1_off"),
    ("mineur", "Fusion bloquée : F5 dit « Rien à calculer : ajouter brutes ou image » alors qu'un canal a son image",
     "Fusion HOO : Ha = image avec fichier, OIII = bloc Image sans fichier. Le bouton Exécuter est bien grisé (« Fusion "
     "bloquée : Ha : requiert Astrométrie ») mais affiche quand même « Exécuter · ≈ 10 min ». F5 ouvre le plan : Exécuter "
     "grisé avec l'infobulle « Rien à calculer : ajouter brutes ou image », et la même phrase dans la fenêtre, alors que Ha "
     "a son image ; la vraie raison (OIII sans fichier) est la ligne d'après.",
     ["Image Ha (fichier) + Image OIII (sans fichier) + Fusion HOO", "survoler Exécuter", "F5"],
     "fusion_nofile_f5_dialog"),
    ("mineur", "Pause sur Gradient 1 : la fenêtre d'exécution dit « Pause vérif après Gradient »",
     "Avec deux Gradient, pause sur le premier : la ligne s'appelle « Gradient 1 », mais la fenêtre dit « Pause vérif après "
     "Gradient » puis « Pause vérif : Gradient · Inspection, puis Continuer », et la barre d'état « Pause vérif à Gradient ». "
     "Le Journal dit aussi « Gradient ajouté » pour le bloc que le graphe appelle « Gradient 1 ».",
     ["Session → Alignement, Empilement, Gradient, Gradient", "Pause vérif sur Gradient 1", "Exécuter"],
     ""),
    ("mineur", "Sélection posée après Alignement : absente de la fenêtre d'exécution",
     "Graphe [Calibration, Alignement, Sélection, Empilement] : la fenêtre d'exécution liste Calibration, Alignement, "
     "Empilement, pas Sélection. Sans Alignement, [Sélection, Empilement, Gradient] a bien sa ligne « Sélection ».",
     ["Session → Calibration, Alignement, Sélection, Empilement", "Exécuter"],
     "selrow_run_end"),
    ("mineur", "À l'ouverture d'un projet Image, toute la palette dit « poser Session/Image d'abord »",
     "Projet avec un bloc Image déjà posé, juste ouvert (aucun bloc choisi) : Gradient, Débruitage, Étoiles/fond, "
     "Astrométrie, Couleur sont grisés avec « poser Session/Image d'abord », alors que l'Image est là. Il faut d'abord "
     "cliquer le bloc Image ; la raison devrait le dire.",
     ["ouvrir un projet dont le flux est un bloc Image seul", "survoler Gradient dans la palette"],
     "edge_cards_at_opening"),
    ("question", "Le dernier bloc d'un flux porte aussi le menu du workflow",
     "Clic droit sur le dernier bloc d'un flux (Gradient 2, Étoiles/fond, Sortie) : en tête Renommer, Dupliquer, "
     "Enregistrer comme modèle…, Supprimer workflow, puis les actions du bloc (Supprimer, Désactiver…). Les autres blocs "
     "n'ont que leurs actions. Deux « Supprimer » de portée différente dans le même menu ; voulu ?",
     ["Image → Gradient, Gradient", "clic droit sur Gradient 1, puis sur Gradient 2"],
     "method_probe"),
]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "anomalies.jsonl").open("w", encoding="utf-8") as fh:
        for sev, title, detail, steps, shot in NOTES:
            fh.write(json.dumps({"area": "engine_shapes", "severity": sev, "title": title, "detail": detail,
                                 "steps": steps, "shot": S(shot) if shot else "", "lang": "fr",
                                 "t": time.strftime("%H:%M:%S")}, ensure_ascii=False) + "\n")
    print(len(NOTES), "notes")
