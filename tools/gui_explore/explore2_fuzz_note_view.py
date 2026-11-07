"""Round 2 fuzz: notes checked by other scripts (each replayed twice there), written here with their screenshot.

    python explore2_fuzz_note_view.py view     # explore2_fuzz_probe2.py: the end of the graph off view
    python explore2_fuzz_note_view.py drop     # explore2_fuzz_targeted.py drop: files from the Explorer refused
"""
import os
import sys
from pathlib import Path

WHAT = sys.argv[1] if len(sys.argv) > 1 else ""
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Explorer  # noqa: E402

ex = Explorer("fuzz", lang="fr")
if WHAT == "view":
    ex.note("gênant", "À l'ouverture, la fin du graphe est hors de la vue (Fusion, Post) en 1366×768",
            "Projet fait avec le modèle SHO, fenêtre en 1366×768 : à l'ouverture le graphe n'est pas ajusté à la vue ; "
            "les trois « Sortie », la Fusion, Recombiner et Post sont à droite, hors de l'écran (barre de défilement en "
            "bas), alors que les blocs visibles sont déjà écrits en tout petit. Même chose avec HOO ; avec « Classique "
            "+ débruitage » seul le bloc Post est coupé. En 1600×950, Post est coupé pour SHO et HOO ; en 1920×1080 "
            "tout tient. « Tout afficher » ramène tout. Reproduit 2/2 à chaque taille.",
            shot="F:/astro_essais/A_REGARDER/exploration2/fuzz/offview_sho_1366.png",
            steps=["fenêtre en 1366×768", "nouveau projet, modèle « SHO »", "regarder le graphe sans rien toucher"])
if WHAT == "drop":
    ex.note("question", "Des brutes glissées depuis l'Explorateur sur une Session sont refusées",
            "Projet d'une session (ou modèle « depuis les poses ») : glisser des fichiers bruts, ou leur dossier, "
            "depuis l'Explorateur de Windows sur le bloc Session du graphe est refusé (curseur interdit, rien "
            "n'est ajouté) ; le panneau de la Session n'a pas non plus de zone où les déposer. Les brutes ne "
            "s'ajoutent que par « Ajouter ▾ ». Les logiciels de référence acceptent ce glisser (import par dépôt). "
            "À juger. Reproduit 2/2.",
            shot="F:/astro_essais/A_REGARDER/exploration2/fuzz/drop_files_on_session.png",
            steps=["nouveau projet, modèle « depuis les poses » (Session 1 vide)",
                   "glisser un dossier de brutes depuis l'Explorateur sur le bloc « Session 1 »"])
if WHAT == "sortie":                                 # explore2_fuzz_probe.py (channel removal ways), 2/2
    import explore2_fuzz as G
    from PySide6.QtCore import Qt
    fz = G.Fuzz2("fr")
    ex = fz.ex
    fz.start({"template": "sho", "lang": "fr"}, "sortie_note")
    fz.activate()
    ex.click_block(next(x for x in fz.items() if fz.title(x) == "Sortie"))
    ex.key(Qt.Key_Delete)
    fz.wait_ready()
    ex.note("mineur", "Bloc « Sortie » d'un canal : Suppr dit « SII gardé : absent du workflow », la palette ne dit rien",
            "Projet SHO : clic sur le bloc « Sortie » du workflow SII puis Suppr : le bloc reste (c'est voulu, le "
            "workflow en a besoin) mais le message dit « SII gardé : absent du workflow », qui n'explique pas pourquoi. "
            "Glisser ce même bloc sur la palette ne fait rien et n'affiche aucun message ; il n'a pas de croix et "
            "son panneau n'a pas de « Supprimer » (cohérent). La règle : un retrait refusé dit sa raison, en mots "
            "simples. Reproduit 2/2.",
            shot=True, steps=["nouveau projet, modèle « SHO »", "clic sur le premier bloc « Sortie » (SII), Suppr",
                              "glisser ce bloc sur la palette"])
if WHAT == "slider":                                 # explore2_fuzz_probe.py (Recombine slider dragged), 2/2
    import explore2_fuzz as G
    fz = G.Fuzz2("fr")
    ex = fz.ex
    fz.start({"template": "sho", "lang": "fr"}, "slider_note")
    ex.click_block(next(x for x in fz.items() if x.node.key == "composition"))
    ex.note("mineur", "Premier réglage d'un Recombiner : deux Ctrl+Z pour l'annuler",
            "Projet SHO : clic sur le bloc Recombiner, puis un de ses curseurs glissé à la souris (ici l'intensité "
            "des étoiles). Ce seul geste crée deux entrées d'annulation : un premier Ctrl+Z remet le curseur à sa "
            "valeur mais laisse un état intermédiaire invisible (le calque de Post enregistré sans réglage), il faut "
            "un second Ctrl+Z pour revenir exactement à l'état d'avant. "
            "Vu par le fuzz (graine 4036), refait avec un vrai glisser de la souris. Reproduit 2/2.",
            shot=True, steps=["nouveau projet, modèle « SHO »", "clic sur le bloc « Recombiner »",
                              "glisser le premier curseur du panneau", "Ctrl+Z"])
if WHAT == "menu":                                   # explore2_fuzz_probe.py: replayed 2/2 on full, classic, nights
    ex.note("gênant", "Clic droit sur Empilement : aucun menu (modèles Complet, Plusieurs nuits…)",
            "Dans un projet fait avec un des modèles qui partent des brutes (vu avec « Complet », « Empilement "
            "classique », « Plusieurs nuits », « Brutes déjà calibrées », « Brutes déjà alignées »), le "
            "clic droit sur le bloc Empilement n'ouvre aucun menu, où qu'on clique sur le bloc. Les autres blocs du "
            "même workflow (Alignement, Astrométrie…) ont leur menu (Supprimer, Désactiver, Inspection, Pause "
            "vérif…). Le bloc a pourtant sa croix et « Supprimer » dans son panneau ; mais Inspection, Pause vérif "
            "et Vers Post ne s'atteignent pas par le menu. Vu aussi par le fuzz (menus vides sur Empilement dans "
            "4 séquences). Reproduit 2/2 sur chaque modèle.",
            shot="F:/astro_essais/A_REGARDER/exploration2/fuzz/empty_menu_full_empilement.png",
            steps=["nouveau projet, modèle « Complet »", "clic droit sur le bloc « Empilement »"])
os._exit(0)
