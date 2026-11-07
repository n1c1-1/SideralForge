"""usages: the anomalies reproduced twice by explore5_usages_*.py, written to anomalies.jsonl (one per title)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

OUT = Path("F:/astro_essais/A_REGARDER/exploration4/usages")
LOG = OUT / "anomalies.jsonl"
seen = {json.loads(x)["title"] for x in LOG.read_text(encoding="utf-8").splitlines() if x.strip()} if LOG.exists() \
    else set()


def note(sev, title, detail, steps, shot, also=()):
    if title in seen:
        return
    seen.add(title)
    rec = {"area": "usages", "severity": sev, "title": title, "detail": detail, "steps": steps,
           "shot": str(OUT / shot) if shot else "", "also": [str(OUT / a) for a in also], "lang": "fr",
           "t": time.strftime("%H:%M:%S")}
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("NOTE", sev, title)


which = sys.argv[1] if len(sys.argv) > 1 else ""

if which == "D":
    note("gênant", "Projet fait depuis un modèle, fermé sans enregistrer : il se rouvre vide et la galerie des modèles "
                   "ne revient pas",
         "Accueil > Nouveau projet > galerie : « Complet » > Créer. Le projet est aussitôt marqué modifié (titre "
         "« FermeP1_1 * ») : son dossier existe déjà sur le disque, mais le workflow du modèle n'y est pas encore. Fermer "
         "la fenêtre tout de suite : « Enregistrer les modifications de « FermeP1_1 » ? » alors que l'utilisateur n'a "
         "rien fait. « Ne pas enregistrer » : le projet reste sur l'Accueil ; rouvert, il est vide (« Aucun workflow », "
         "« + Départ : Session/Image ») et la fenêtre des modèles ne s'ouvre plus (elle n'apparaît qu'à la création) : "
         "le modèle choisi est perdu sans moyen de le reprendre. Vu deux fois (FermeP1_1, FermeP1_2).",
         ["Accueil > Nouveau projet « FermeP1 », Créer", "galerie : Complet > Créer",
          "fermer la fenêtre aussitôt > Ne pas enregistrer", "relancer, rouvrir le projet depuis l'Accueil"],
         "P1_2_template_not_saved_reopened.png")
    note("gênant", "Enregistrer sous… annulé : le projet d'origine est quand même enregistré",
         "Projet enregistré, puis des modifications (une branche Couleur, l'Empilement vers Post) : titre avec « * ». "
         "Fichier > Enregistrer sous…, puis Annuler dans le choix du dossier : rien ne devrait changer, mais le "
         "project.json d'origine a été réécrit avec les modifications et le « * » a disparu. Les modifications que "
         "l'utilisateur gardait en suspens sont enregistrées sans qu'il l'ait demandé. Vu deux fois.",
         ["projet enregistré", "une modification (titre « * »)", "Fichier > Enregistrer sous…", "Annuler le dossier",
          "le titre n'a plus de « * » ; le projet d'origine contient la modification"], "S_1_saved_as.png")
    note("gênant", "Enregistrer sous… écrit aussi les modifications dans le projet d'origine",
         "Projet « SousS_1 » enregistré, puis Débruitage posé (non enregistré). Fichier > Enregistrer sous… > un dossier "
         "neuf : la copie est ouverte, mais le projet d'origine a lui aussi reçu le Débruitage. Dans les logiciels de "
         "référence, Enregistrer sous laisse l'original tel qu'il était au dernier enregistrement. Vu deux fois.",
         ["projet enregistré", "Débruitage posé (titre « * »)", "Fichier > Enregistrer sous… > dossier neuf",
          "rouvrir l'original : le Débruitage y est"], "S_2_saved_as.png")
    note("question", "Enregistrer sous… : la copie garde le nom de l'original",
         "Après Enregistrer sous… dans « SousS_1_copie », la fenêtre affiche toujours « SousS_1 » (titre, en-tête) et le "
         "fichier de la copie s'appelle « SousS_1 » : l'Accueil montre deux projets du même nom, rien ne dit lequel est "
         "ouvert. Ailleurs (photo), le document prend le nom choisi à l'enregistrement. Vu deux fois.",
         ["projet « SousS_1 »", "Fichier > Enregistrer sous… > dossier « SousS_1_copie »", "regarder le titre, l'Accueil"],
         "S_2_saved_as.png")
    note("gênant", "Dossier des brutes renommé hors de l'application : le panneau dit « lights · 0 fichier », pas qu'il est "
                   "introuvable",
         "Projet enregistré et fermé, le dossier des brutes renommé dans l'Explorateur, projet rouvert. La barre d'état "
         "et Exécuter (grisé) le disent bien (« dossier des brutes introuvable … (son disque est-il branché ?) »), le bloc "
         "Session aussi (« dossier des brutes i… », coupé). Mais le panneau de la Session, là où on corrige, montre "
         "« Brutes · 0 / 0 utilisées » et « lights · 0 fichier » comme un dossier vide : aucun mot « introuvable », "
         "aucun moyen de lui donner le nouvel emplacement (il faut deviner : × puis Ajouter > dossier). Vu deux fois.",
         ["projet d'une Session (un dossier de brutes), Ctrl+S, fermer", "renommer le dossier des brutes",
          "rouvrir le projet, clic sur la Session"], "O_1_folder_renamed.png")

if which == "E":
    note("gênant", "Gradient 2 : aucun réglage dans son panneau (Gradient 1 a « Degré de la surface » et « Modifier "
                   "échantillons… »)",
         "Projet Vierge, Image (TIFF 16 bits), Astrométrie, Gradient, Gradient, Couleur. Clic sur Gradient 1 : panneau "
         "avec Méthode, « Degré de la surface », « Modifier échantillons… », « Réglages avancés (1) ». Clic sur "
         "Gradient 2 (même méthode Polynôme) : seulement les cartes Méthode, Supprimer, Désactiver ; aucun réglage, pas "
         "d'échantillons, pas de réglages avancés. Le second Gradient ne peut pas être réglé. Vu deux fois, avant et "
         "après réouverture.",
         ["Accueil > Nouveau projet > Vierge", "Image posée, Choisir image… > un TIFF",
          "double-clic Astrométrie, Gradient, Gradient, Couleur", "clic sur Gradient 1 puis sur Gradient 2"],
         "E_I1_reopened.png")
    note("mineur", "Une seule Image, nommée « Image 1 », dès qu'une image de la chaîne part vers Post",
         "Image → Astrométrie → Gradient 1 → Gradient 2 → Couleur. « Vers Post » sur Gradient 1 (calque « Image · "
         "Gradient ») : le bloc Image, toujours seul à l'écran, devient « Image 1 » (et le reste après réouverture, "
         "et en anglais). Il n'y a pas d'« Image 2 » : le numéro laisse croire à une deuxième image. Vu deux fois.",
         ["Image (TIFF) + Astrométrie, Gradient, Gradient, Couleur", "clic droit sur Gradient 1 > Vers Post",
          "regarder le bloc Image"], "E_I1_reopened.png")
    note("question", "« Tout afficher » n'est pas gardé : le projet rouvert revient à la taille lisible",
         "« Tout afficher » activé, Ctrl+S, fenêtre fermée, projet rouvert dans une nouvelle fenêtre : le bouton est "
         "désactivé, le graphe revient à la taille lisible avec défilement. Le bloc choisi, lui, est gardé. Faut-il "
         "garder ce choix d'affichage (par projet, ou pour tous) ? Vu deux fois.",
         ["un projet", "clic sur « Tout afficher »", "Ctrl+S", "fermer la fenêtre, rouvrir le projet"],
         "E_I2_reopened.png")
    note("question", "Projet fait en français, rouvert en anglais : le workflow garde le nom français du modèle",
         "Projet créé avec le modèle « Complet » dans l'interface en français, enregistré ; l'application relancée en "
         "anglais : le workflow s'appelle toujours « Complet » (les blocs, eux, sont en anglais : Registration, "
         "Stacking…). De même les branches « Workflow · Empilement brut ». C'est un nom donné au projet (normal qu'il "
         "reste), mais l'utilisateur ne l'a pas choisi : à décider. Vu deux fois.",
         ["interface en français", "Accueil > Nouveau projet > Complet", "Ctrl+S, fermer",
          "relancer en anglais, rouvrir le projet"], "E_L2_english.png")

if which == "F":
    note("mineur", "Projet rouvert : Recombiner était choisi, il s'ouvre sur la Session 1",
         "Modèle Complet, Recombiner choisi (son mode de fusion et un curseur changés), Ctrl+S, fenêtre fermée, projet "
         "rouvert dans une nouvelle fenêtre : le bloc choisi est la Session 1, pas Recombiner (les autres blocs essayés "
         "— Couleur, Gradient 2, Fusion — sont bien repris). Vu deux fois.",
         ["Accueil > Nouveau projet > Complet", "clic sur Recombiner, Mode de fusion des étoiles changé",
          "Ctrl+S, fermer la fenêtre", "rouvrir le projet"], "F_2_reopened.png", also=["F_2_built.png"])

if which == "G":
    note("gênant", "Nouveau projet proposé dans le dossier des darks qu'on vient d'ajouter",
         "Un premier projet : Session, Brutes > Ajouter dossier…, Darks > Ajouter dossier…, Ctrl+S, fermer. Accueil > "
         "Nouveau projet… : le Dossier proposé est <données>/Ou_2/darks/Second_2, c'est-à-dire DANS le dossier des darks "
         "(le dernier dossier ouvert), au lieu du dossier des projets. Le chemin est coupé au milieu (« …2o/data/Ou_2/"
         "darks/Second_2 ») : on ne le remarque pas, Créer écrit le projet au milieu des fichiers de la caméra. "
         "Import dossier évite déjà ce cas ; Nouveau projet non. Vu trois fois.",
         ["un projet : Session, Brutes > Ajouter dossier…, Darks > Ajouter dossier…", "Ctrl+S, fermer le projet",
          "Accueil > Nouveau projet… : lire le Dossier"], "H2_new_project_dialog.png")
    note("gênant", "Deux sessions peuvent porter le même nom (« Session 2 ») : le nom par défaut ne tient pas compte des "
                   "noms donnés",
         "Projet de deux flux (Session 1, Session 2), Session 2 supprimée, Ctrl+S, rouvert ; la Session 1 renommée "
         "« Session 2 » (accepté), Ctrl+S, rouvert ; une Session posée sur « + Workflow » : elle s'appelle aussi "
         "« Session 2 ». Deux blocs « Session 2 », deux noms identiques dans les menus de partage des darks, alors que "
         "renommer en un nom déjà pris est refusé (décision 8). Vu deux fois.",
         ["projet Vierge, Session + brutes, Calibration…", "une Session sur « + Workflow » (Session 2)",
          "clic droit Session 2 > Supprimer", "Ctrl+S, fermer, rouvrir", "clic droit Session 1 > Renommer session > "
          "« Session 2 »", "Ctrl+S, fermer, rouvrir", "glisser Session sur « + Workflow »"], "G_N2_names.png")
    note("gênant", "Dossier des darks disparu : seuls la barre d'état et Exécuter le disent ; Session et Calibration "
                   "affichent « 0 dark »",
         "Projet enregistré avec 6 brutes et un dossier de 4 darks ; le dossier des darks renommé dehors, projet rouvert. "
         "Barre d'état et infobulle d'Exécuter : « dossier des darks introuvable … ». Mais le bloc Session dit "
         "« 5 brutes · 0 dark · 0 flat », la Calibration « 0 dark · 0 flat · 0 offset » (son infobulle ne dit rien), "
         "le panneau « Darks · aucun » et « 📁 darks · 0 fichier » : on lit qu'il n'y a pas de darks, pas qu'ils sont "
         "introuvables (pour un dossier de brutes disparu, le bloc Session le dit au moins). Une brute retirée du "
         "dossier : 6 → 5 sans un mot, ce qui est attendu pour un dossier. Vu deux fois.",
         ["projet : Session, brutes (6), Darks > Ajouter dossier (4)", "Ctrl+S, fermer",
          "renommer le dossier des darks dans l'Explorateur", "rouvrir : regarder Session, Calibration, panneau"],
         "G_X2_darks_gone.png")

if which == "I":
    note("mineur", "Fusion passée en HSO : le workflow et le bloc s'appellent toujours « SHO (palette Hubble) »",
         "Projet SHO fait depuis l'Accueil ; clic sur la Fusion, Palette : HSO. Le bouton HSO est bien choisi (et "
         "gardé après Ctrl+S, fermer, rouvrir), mais le titre du panneau, le nom du workflow (palette « pour « SHO "
         "(palette Hubble) » ») et le résumé du bloc (« SHO (palette Hubble) · HSO · SII, Ha, OIII ») disent SHO et HSO "
         "à la fois ; le calque de Post portera ce nom. Vu deux fois.",
         ["Accueil > Nouveau projet > SHO", "clic sur Fusion > Palette : HSO", "Ctrl+S, fermer, rouvrir"],
         "I2_reopened.png")

if which == "K":
    note("question", "Modèle Complet (ou Plusieurs nuits) : le bloc Recombiner n'apparaît qu'une fois les brutes ajoutées",
         "Projet créé avec « Complet » : le graphe s'arrête à Étoiles/fond, la colonne Composition est vide (le texte du "
         "modèle annonce l'image sans étoiles et les étoiles dans Post). Brutes > Ajouter dossier… : un bloc Recombiner "
         "apparaît de lui-même dans Composition, sans message ; le workflow enregistré, lui, ne change pas. Même chose "
         "avec « Plusieurs nuits » ; pas avec « Empilement classique » (sans Étoiles/fond). Le graphe change de forme "
         "quand on ajoute des fichiers : est-ce voulu ? Vu deux fois.",
         ["Accueil > Nouveau projet > Complet", "regarder la colonne Composition", "Session : Brutes > Ajouter dossier…",
          "regarder à nouveau"], "K_full_avant_apres_brutes.png")
