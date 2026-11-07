"""gros_projets: the anomalies measured by explore5_gros_projets_measure.py (S, M, L, XL; every measure twice), written
to exploration4/gros_projets/anomalies.jsonl in the harness's format (no window: the screenshots are the runs')."""
from __future__ import annotations

import json
import time
from pathlib import Path

OUT = Path("F:/astro_essais/A_REGARDER/exploration4/gros_projets")
SIZES = ("baseline S = 1 session de 15 brutes, 1 workflow ; M = 4 sessions de 150, 9 workflows ; L = 16 sessions "
         "(7 800 brutes, dont une de 3 000), 27 workflows avec branches, 3 Fusion, 15 filtres dans Post ; XL = 24 "
         "sessions (14 200 brutes, une de 5 000), 44 workflows, 3 Fusion, 25 filtres")


def M(size):
    f = OUT / f"measures_{size}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def fr(size, gesture):
    v = [r["freeze"] for r in M(size) if r["gesture"].startswith(gesture)]
    return "/".join(f"{x:.2f}".replace(".", ",") for x in v[:2]) + " s" if v else "—"


def note(sev, title, detail, shot, steps):
    rec = {"area": "gros_projets", "severity": sev, "title": title, "detail": detail, "steps": steps,
           "shot": str(OUT / shot), "lang": "fr", "t": time.strftime("%H:%M:%S")}
    with (OUT / "anomalies.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(sev, title)


def row(g):
    return " ; ".join(f"{s} {fr(s, g)}" for s in ("S", "M", "L", "XL") if M(s))


BUILD = ["projet construit par explore5_gros_projets_measure.py <taille> (sessions de fausses brutes minuscules, "
         "workflows, branches, Fusion)", "fermer puis rouvrir le projet"]
note("bloquant", "Gros projet : l'ouverture fige la fenêtre de 9 s (27 workflows) à 40 s (44 workflows)",
     f"Fenêtre bloquée (temps passé sans répondre, mesuré deux fois) : {row('open project')}. Le travail de fond "
     "(estimations, vérification des brutes) dure ensuite 26 s en L et 84–90 s en XL, pendant lesquelles la fenêtre répond par à-coups. "
     "Machine chargée par un entraînement : comparer les rapports (L ≈ 13 × S, XL ≈ 60 × S : la croissance est plus que proportionnelle au nombre de workflows). Profil (profile_L.txt) : les "
     "estimations de durée relisent et retrient la liste des brutes de toutes les sessions pour chaque workflow et chaque "
     "étape (store.frames appelé ~2 200 fois, 7,9 millions de comparaisons de chemins) dans le fil de fond, qui garde "
     "le verrou de Python ; la vérification des sessions ouvre aussi l'en-tête de chaque brute tant qu'aucune taille "
     "n'est lue (avec mes fausses brutes illisibles, ce dernier point est plus lourd que sur de vraies). " + SIZES,
     "L_01_open.png", BUILD)
note("gênant", "Gros projet : chaque modification du graphe fige la fenêtre plus d'une seconde",
     "Fenêtre bloquée, deux mesures chacune — poser un bloc (double-clic palette) : " + row("palette double-click") +
     " ; Suppr : " + row("Del") + " ; Ctrl+Z : " + row("Ctrl+Z (back") + " ; Ctrl+Y : " + row("Ctrl+Y") +
     " ; renommer une session (la mise à jour seule) : " + row("rename session (model") + ". Croît à peu près "
     "comme le nombre de workflows (S → L : × 10 à × 13 ; L → XL, 27 → 44 workflows : × 3). Profil : à chaque modification le bouton Exécuter recalcule "
     "le plan de tous les workflows (update_run_button → plan_all, ~1,3 s sous profileur) dans le fil de la fenêtre, "
     "puis le graphe est reconstruit, puis les estimations repartent en fond. " + SIZES,
     "L_03_session_panel.png", BUILD + ["sélectionner le Débruitage de Couleur 1", "double-clic sur Gradient dans la "
                                        "palette ; Suppr ; Ctrl+Z ; Ctrl+Y"])
note("gênant", "Gros projet : glisser un bloc de la palette sur une place prend 1,5 s (L) à 4,8 s (XL)",
     "Lâcher Débruitage sur une place : " + row("drag palette") + " (le survol qui allume les places : 0,13–0,20 s en "
     "L, 0,30 s en XL). Ctrl+Z de ce dépôt : " + row("Ctrl+Z (drag)") + ". " + SIZES, "L_07_drag_ghosts_1.png",
     BUILD + ["clic sur un Empilement", "glisser Débruitage de la palette sur la première place allumée"])
note("gênant", "Gros projet : F5 met 1 s (L) à 3,5 s (XL) à ouvrir la fenêtre d'exécution",
     "F5 jusqu'à la fenêtre d'exécution affichée : " + row("F5") + " (72 lignes d'étapes en L). " + SIZES,
     "L_04_run_window.png", BUILD + ["F5"])
note("mineur", "Gros projet : la liste « Non démarrés » de la fenêtre d'exécution est coupée",
     "Quand l'exécution ne démarre pas (ici : bloquée par l'exploration), l'encadré rouge liste les 72 étapes non "
     "démarrées en un seul paragraphe ; en L il dépasse l'encadré et s'arrête au milieu d'un nom (« Alignement "
     "(Couleur 3, Couleur 3 · »), sans défilement ni infobulle. En M (30 étapes) il tient (M_04_run_window.png). "
     "Vu aux deux F5.", "L_04_run_window.png", BUILD + ["F5"])
note("mineur", "Gros graphe : le bloc Post passe sous la barre de défilement verticale",
     "Dès que le graphe défile verticalement (L : scène 1 460 × 2 348 à l'échelle 0,75), le bord droit du bloc Post "
     "est caché sous la barre de défilement de la vue (voir L_01_crop_right.png, aussi L_03 et L_06 à 1366×768). "
     "En S et M, sans barre, il est entier.", "L_01_crop_right.png", BUILD)
note("question", "Pendant un glisser dans un gros graphe, 116 places s'allument partout",
     "Glisser Débruitage depuis la palette avec 27 workflows : 116 places en pointillé (« Débruitage », « nouvelle "
     "branche ») couvrent tout le graphe, y compris hors de l'écran et sous des fils ; certaines sont coupées au bord "
     "droit par le panneau. S : 10–14, M : 49, XL : 197. Faut-il n'allumer que le flux choisi (ou proche du pointeur) ?",
     "L_07_drag_ghosts_1.png", BUILD + ["clic sur un Empilement", "glisser Débruitage depuis la palette, sans lâcher"])
note("question", "Gros projet : les sessions sont empilées à gauche, loin de leur chaîne",
     "Avec 16 sessions, la colonne des Session/Calibration est serrée à gauche tandis que chaque workflow est à sa "
     "propre hauteur : le fil de « Couleur nuit 4 » descend d'environ 250 px jusqu'à son Alignement, ceux des 4 "
     "sessions Ha montent tous vers un Alignement placé au-dessus d'elles. Pas de chevauchement de blocs (vérifié en "
     "S, M, L), mais il faut suivre de longs fils. Aligner chaque groupe de sessions sur sa chaîne ?",
     "L_03_session_panel.png", BUILD)
note("mineur", "Branche nommée « SII · Gradient 2 2 » : numéro doublé",
     "Dans le projet XL (3 branches par workflow), les branches Gradient d'un même workflow s'appellent « SII · "
     "Gradient », « SII · Gradient 2 » et « SII · Gradient 2 2 » (sous-titre du bloc « Gradient 2 », tous les "
     "workflows Ha, OIII, SII, Couleur 1…6 : vu sur 9 workflows). Le nom unique ajoute « 2 » à un nom qui finit déjà "
     "par le numéro de la copie. Construit par l'appel du modèle qu'utilise le dépôt d'un bloc sous un bloc "
     "(add_branch).", "XL_01_open.png",
     ["projet XL de explore5_gros_projets_measure.py", "un workflow SII : branche Gradient depuis Empilement, puis deux "
      "branches Gradient depuis Astrométrie", "lire le sous-titre des blocs Gradient 2"])
