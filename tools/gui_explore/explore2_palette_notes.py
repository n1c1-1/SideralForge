"""palette_matrix round 2: the anomalies found (each reproduced twice by the explore2_palette_* scripts), noted in plain
French with their screenshot. A few screenshots are taken again here at the moment that shows the anomaly."""
from __future__ import annotations

from pathlib import Path

from explore2_palette_lib import Matrix

from PySide6.QtCore import QEvent, QPointF
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QApplication

m = Matrix("notes")
ex = m.ex
D = ex.out
RAW = D / "raw" / "shots"


def hover_shot(kind, card, name, click=None):
    ex.open(ex.project(kind))
    ex.settle(8)
    if click:
        ex.click_block(ex.block(click))
    c = ex.card(card)
    p = QPointF(c.width() / 2, c.height() / 2)
    QApplication.sendEvent(c, QEnterEvent(p, c.mapToGlobal(p), c.mapToGlobal(p)))
    ex.settle(3)
    f = ex.shot(name)
    QApplication.sendEvent(c, QEvent(QEvent.Leave))
    return f


f_open_image = hover_shot("image", "gradient", "n_image_ouverture_tout_grise")
f_stars = hover_shot("session_empty_wf", "stars", "n_etoiles_fond_raison_sortie")
f_accord = hover_shot("legacy", "selection", "n_un_seul_selection")
f_elision = hover_shot("image", "registration", "n_pas_de_alignement", click="image")

# one Fusion on a mono project whose channels have no Astrometry
ex.open(ex.project("mono"))
ex.settle(8)
fus_reason = ex.card("fusion").reason
ex.double_click_card("fusion")
ex.settle(6)
it = ex.block("fusion")
ex.hover(ex.view.viewport(), ex.point_of(it))
f_fusion = ex.shot("n_fusion_sans_astrometrie")
run_tip = ex.w.bt_run.toolTip()
blocks_ha = [n["block"] for n in ex.s.graph(ex.s.paths[0])]

S = str
notes = [
    ("bloquant", "Pause vérif sur une copie (Gradient 2, Couleur 2, Débruitage 3) : Exécuter répond « Rien à exécuter »",
     "Workflow avec un bloc posé deux fois (Image → Astrométrie → Gradient 1 → Gradient 2 → Couleur 1 → Couleur 2). "
     "Sans pause, Exécuter liste bien Image, Astrométrie, Gradient 1, Gradient 2, Couleur 1, Couleur 2. Avec « Pause "
     "vérif » sur Gradient 1, la fenêtre s'arrête à Gradient 1 comme prévu. Mais avec « Pause vérif » sur Gradient 2 (ou "
     "Couleur 2, ou Débruitage 3 dans un workflow complet), la fenêtre d'exécution est vide et dit « Rien à exécuter », "
     "alors que le bouton annonce « Exécuter · ≈ 9 min » et que rien n'est calculé : le projet ne peut plus être "
     "exécuté du tout tant que la pause reste sur la copie, et rien ne dit pourquoi. Vu aussi dans un workflow complet "
     "à partir de brutes (pause sur Gradient 3, Débruitage 3 ou Couleur 3). Attendu (PRINCIPLES : les copies sont des blocs comme les autres ; pause pour vérif) : l'exécution "
     "s'arrête après Gradient 2.",
     S(D / "probe_run_fr_a2_pause_fond2.png"),
     ["projet Image", "double-clic Astrométrie, Gradient, Gradient", "clic droit sur Gradient 2 > Pause vérif",
      "Exécuter"]),
    ("gênant", "Projet Image seule : à l'ouverture la palette grise tout (« poser Session/Image d'abord ») alors que "
     "l'Image est posée",
     "Projet avec un seul workflow à partir d'une image (bloc Image → Post, aucun autre bloc). À l'ouverture, et après "
     "un clic dans le vide du graphe : l'en-tête de la palette dit « nouveau workflow » au lieu de « pour « Image » », "
     "toutes les cartes de Calibration à Étoiles/fond sont grisées avec « Poser Session/Image d'abord », et le panneau "
     "de droite montre « Images / Importer dossier… », le panneau des brutes, dans un projet sans brutes. Un clic sur "
     "le bloc Image remet tout d'aplomb (Astrométrie et Gradient disponibles). Même chose en anglais (« place "
     "Session/Image first »). Attendu (PRINCIPLES : un bloc grisé dit pourquoi, et juste) : la palette vise le "
     "workflow Image, Astrométrie et Gradient disponibles.", f_open_image,
     ["ouvrir un projet Image seule", "survoler la carte Gradient"]),
    ("gênant", "Une seconde Fusion posée ne se voit pas : un seul bloc Fusion dessiné pour deux",
     "Projet mono (Ha, OIII). Double-clic sur la carte Fusion : une Fusion « HOO » apparaît. Second double-clic (ou "
     "Fusion lâchée sur la zone « + Flux ») : un workflow « HOO 2 » est créé (4 workflows, la palette dit « pour « HOO "
     "2 » »), mais le graphe ne montre toujours qu'un seul bloc Fusion, « HOO · Ha, OIII », dont le panneau s'appelle "
     "« HOO » ; seuls deux blocs Recombiner trahissent la seconde. On ne peut pas choisir, régler ou supprimer la "
     "seconde Fusion seule. Attendu (PRINCIPLES : un bloc posé se voit ; les copies sont numérotées, même nom partout) : "
     "deux blocs Fusion distincts.", S(D / "fusion2_dbl.png"),
     ["projet mono Ha + OIII", "double-clic carte Fusion", "double-clic carte Fusion"]),
    ("gênant", "Fusion posée sur des canaux sans Astrométrie : la carte n'est pas grisée, le bloc arrive en rouge et "
     "Exécuter se grise",
     "Projet mono (Ha, OIII : Calibration → Alignement → Empilement chacun). La carte Fusion est disponible (raison : "
     f"« {fus_reason} »). Double-clic : la Fusion est posée mais aussitôt bordée de rouge, son infobulle dit « Ne peut "
     "pas s'exécuter : Ha : requiert Astrométrie », et le bouton Exécuter est grisé (« " + run_tip + " »). Attendu "
     "(PRINCIPLES : un bloc qui a besoin d'un bloc loin en amont est refusé avec la raison) : la carte grisée avec "
     "« il faut l'Astrométrie dans chaque canal », ou l'Astrométrie posée avec elle.", f_fusion,
     ["projet mono Ha + OIII", "double-clic carte Fusion", "survoler la Fusion, regarder Exécuter"]),
    ("question", "Poser une Fusion ajoute Débruitage et Étoiles/fond dans les workflows des canaux",
     "Projet mono (Ha, OIII : Calibration → Alignement → Empilement). Après un double-clic sur Fusion, les workflows "
     f"Ha et OIII ont en plus Débruitage et Étoiles/fond (Ha : {blocks_ha}), et deux blocs Recombiner apparaissent, "
     "sans que l'utilisateur les ait posés ni qu'un message le dise (le texte du panneau l'explique : « les étoiles "
     "sont d'abord retirées dans chaque canal »). Un seul Ctrl+Z retire bien le tout. À trancher : ajout automatique "
     "voulu, ou un message qui le dit au moment de la pose.", S(D / "fusion_deux_fois.png"),
     ["projet mono Ha + OIII", "double-clic carte Fusion"]),
    ("mineur", "La raison de la carte Étoiles/fond parle de « Sortie »",
     "Workflow de brutes sans Empilement : la carte Étoiles/fond est grisée avec « Sortie requiert Empilement avant » "
     "(en anglais « Output needs Stacking first »). La carte s'appelle Étoiles/fond ; « Sortie » n'est pas un bloc à "
     "l'écran. Attendu (PRINCIPLES : même nom partout ; la Sortie n'est pas un bloc à l'écran) : « Étoiles/fond requiert "
     "Empilement avant ».", f_stars,
     ["projet Session (workflow vide)", "survoler la carte Étoiles/fond"]),
    ("mineur", "Raisons de la palette : accords et élisions fautifs en français",
     "Workflow complet : « Un seul Calibration par workflow », « Un seul Sélection… », « Un seul Astrométrie… » "
     "(féminins : « Une seule Calibration »). Workflow Image : « Pas de Alignement depuis fichier image », « Pas de "
     "Empilement… » ; Image étirée : « pas de Alignement / Astrométrie / Empilement depuis image étirée » (« d'Alignement »). "
     "Attendu (PRINCIPLES : traduction française complète) : un français correct.", f_accord,
     ["projet workflow complet", "survoler la carte Sélection", "projet Image, clic sur l'Image",
      "survoler la carte Alignement"]),
]
for sev, title, detail, shot, steps in notes:
    ex.note(sev, title, detail, shot=shot if shot and Path(shot).is_file() else True, steps=steps)
print("elision shot", f_elision)
ex.close()
