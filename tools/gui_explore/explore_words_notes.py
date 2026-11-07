"""Words and layout: the anomalies, written after a look at the walks (walk_*.json, suspect_strings.txt,
layout_groups.txt, the probes). Each was reproduced by a walk or a probe of this folder; the image is in notes/.
Rewrites <out>/anomalies.jsonl (this area's only).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

OUT = Path("F:/astro_essais/A_REGARDER/exploration/words_layout")
N = OUT / "notes"


def rec(severity, title, detail, shot, steps, lang="fr"):
    return {"area": "words_layout", "severity": severity, "title": title, "detail": detail, "steps": steps,
            "shot": str(N / shot) if shot else "", "lang": lang, "t": time.strftime("%H:%M:%S")}


NOTES = [
    rec("gênant", "À 1366×768, l'Inspection élargit la fenêtre au-delà de l'écran, et elle le reste",
        "Le bandeau des blocs de l'Inspection impose sa largeur minimale à la fenêtre : 1461 px en français pour un "
        "workflow complet (« Full » : Images, Calibration … Split étoiles et fond, Sortie, Post-traitement), 1567 px "
        "avec la chaîne complète (Gradient 1/2, Débruitage 1/2) ; 1369 / 1457 px en anglais. De retour sur le Graphe la "
        "fenêtre garde cette largeur et refuse d'être ramenée à 1366 (la largeur minimale reste 1461). Sur un écran "
        "1366, la partie droite (bouton Exécuter, panneau du bloc) sort de l'écran. Inspection de Gradient : 1458 px.",
        "03_inspect_fenetre_1461px.png",
        ["Fenêtre 1366×768, projet avec le workflow « Full »", "Double-clic sur Split étoiles et fond (ou Gradient)",
         "La fenêtre passe à 1461 px ; Graphe : elle reste à 1461 px", "explore_words_width.py fr"]),
    rec("gênant", "Journal : lignes en anglais et clés internes (« Block fond~2 switched off »)",
        "En français, le Journal montre « Block fond~2 switched off », « Block fond switched on », « Branch "
        "\"Workflow · Débruitage\" made » ; en anglais aussi « Block fond switched off » (la clé interne du bloc au lieu "
        "de son nom, « fond~2 » au lieu de « Gradient 2 »). Quatre autres lignes du Journal n'ont pas d'entrée "
        "française (lecture du code) : « \"{name}\" sent to Post », « Blank page: the workflow is built from the "
        "palette », « Branch \"{name}\" made », « Template \"{name}\" saved ».",
        "02_journal_anglais_cles.png",
        ["Projet avec Gradient ×2", "Menu de Gradient 2 > Désactiver ; déposer un Débruitage sur « nouvelle branche »",
         "Affichage > Journal"]),
    rec("gênant", "« Split » : un mot anglais dans le nom français du bloc « Split étoiles et fond »",
        "Le bloc s'appelle « Split étoiles et fond » partout en français (palette, graphe, panneau, Inspection, menus, "
        "Journal, raisons des blocs grisés : « Il lui faut un bloc Split étoiles et fond dans ce workflow », « seul "
        "Split étoiles et fond vient après le bloc Sortie »), et « le bloc Split » dans une dizaine d'infobulles "
        "(« Combine les deux images du bloc Split », « Le bloc Split n'est pas encore calculé »). L'aperçu des modèles "
        "l'appelle « Étoiles ». ~25 textes (liste dans suspect_strings.txt).",
        "08_split_anglais.png", ["Graphe d'un projet « Full », en français"]),
    rec("gênant", "Galerie des modèles : deux descriptions en anglais, « Colour » dans un aperçu",
        "Onglet Filtres, en français : la description de « SHO (palette Hubble) » (« Narrowband: one workflow per "
        "filter… ») et celle de « Bi-bande (capteur couleur) » (« A colour camera behind a dual-band filter… ») sont "
        "en anglais (pas d'entrée dans le catalogue). Dans l'aperçu Bi-bande, le premier bloc est écrit « Colour ».",
        "05_galerie_texte_anglais.png",
        ["Nouveau projet > fenêtre des modèles > onglet Filtres", "Cliquer SHO puis Bi-bande (aperçu : 06_apercu_colour.png)"]),
    rec("gênant", "Graphe : « + Session » et « + Nouveau flux » l'un sur l'autre (texte sur texte)",
        "Avec deux sessions, la place « + Session » (sous la dernière session) et la zone « + Nouveau flux : déposez "
        "une Session ou une Image ici » se recouvrent sur 32 px : on lit « + Session + Nouveau flux… ». Recouvrement de "
        "13 px avec une session et un workflow vide (session_empty_wf, jpg_mistake). La zone du nouveau flux mord "
        "aussi sur le bas des blocs Session 2 / Image (session+image). Pendant le glisser d'un bloc image, les places "
        "« ⑂ nouvelle branche » recouvrent la zone du nouveau flux (108×9 px). Aux deux tailles, dans les deux langues.",
        "01_places_superposees.png",
        ["Projet à deux sessions (two_sessions), Graphe", "Regarder sous Session 2"]),
    rec("gênant", "Infobulle de « Exécuter jusqu'ici » : clés internes et nom du workflow parmi les blocs",
        "Dans l'Inspection d'un bloc non calculé, l'infobulle du bouton « Exécuter jusqu'ici » liste les blocs à "
        "calculer avec des clés internes : « Calibration, Alignement, Empilement, Astrométrie, Gradient, Débruitage, "
        "denoise~2, Couleur, Workflow, Split étoiles et fond » ; avec deux sessions : « calibration@1, calibration@2, "
        "Alignement ». Le nom du workflow (« Workflow », « Full », « HOO », « Empilement brut ») y figure comme un bloc "
        "(c'est la Sortie). Le premier Gradient n'y est pas numéroté. Mêmes clés en anglais.",
        "12_inspect_gradient2.png",
        ["Chaîne complète avec Débruitage ×2 et deux sessions", "Double-clic sur Split étoiles et fond (ou Sélection)",
         "Survoler « Exécuter jusqu'ici »"]),
    rec("gênant", "À 1366×768 le graphe d'une chaîne complète est illisible (0,60 : titres ≈ 6 pt) et défile",
        "La chaîne complète (Session → Split) est dessinée à l'échelle minimale 0,60 : titres ≈ 6 pt, résumés ≈ 5 pt, "
        "et 260 à 470 px restent cachés à droite (barre de défilement). À 1920×1080 déjà 0,64 à 0,69. Les résumés sont "
        "coupés sur presque chaque bloc (« 15 brutes · 0 darks, … », « drizzle ×1,5 · rejet … », « toutes les images, "
        "… », « étoiles Gaia · balan… » ; l'infobulle les donne en entier). Avec le Journal ouvert (200 px fixes), le "
        "graphe n'a plus que 468 px de haut.",
        "07_graphe_1366_illisible.png", ["Fenêtre 1366×768, chaîne complète construite depuis la palette"]),
    rec("gênant", "En anglais, « & » est pris pour un raccourci clavier : « Split stars _background »",
        "Les boutons dont le texte contient « & » le perdent et soulignent l'espace suivant : bandeau de l'Inspection "
        "« Split stars _background », bouton du panneau « New Combine stars _background layers block ». Les actions "
        "de menu ayant le même texte (« + Recombine ») sont touchées de la même "
        "façon (Qt, à vérifier à l'écran). Le français n'a pas le problème (« et »).",
        "04_en_esperluette_mangee.png", ["Interface en anglais", "Inspection de Stars/background"], lang="en"),
    rec("mineur", "Un même objet, plusieurs noms : Post / Post-traitement, Inspect / Inspection, Coupé / Désactivé…",
        "Post : « Post-traitement » (onglet, bloc, Affichage) mais « Vers le Post » (palette, boutons du panneau Split), "
        "« Envoyer son image au Post » et « Envoyer l'image sans étoiles à Post-traitement » dans le même menu, « dans "
        "le Post ». Inspection : « Ouvre le bloc dans Inspect » (bouton Modifier les échantillons). Désactiver : "
        "l'infobulle d'un bloc désactivé dit « Coupé : l'image passe. D ou le menu le rétablit. ». Flux / workflow : "
        "« Nouveau flux », un flux parallèle nommé « Flux », « La Fusion combine des flux… », partout ailleurs "
        "« workflow ». RVB (galerie : « RVB avec filtres », « LRVB ») / RGB, LRGB (boutons de la Fusion). Exécuter : "
        "« Où le lancement calcule » (combo GPU). Poses / brutes / images pour « frames » (« À partir de brutes », "
        "« SUR LES POSES », bandeau « Images » ; le catalogue a deux valeurs pour « frames » : « images » et « de "
        "poses »).",
        "13_graphe_noms.png", ["Chaîne complète en français : palette, graphe, menus des blocs, panneau Split"]),
    rec("mineur", "Copies numérotées : « Gradient 1 » dans le graphe, « Gradient » ailleurs",
        "Graphe : « Gradient 1 », « Gradient 2 », « Débruitage 1 », « Débruitage 2 ». Bandeau de l'Inspection : "
        "« Gradient », « Gradient 2 », « Débruitage », « Débruitage 2 » ; titre « Gradient · Workflow » pour le premier ; "
        "Journal « Gradient ajouté » ; fenêtre d'exécution « Gradient 1 ». Dans l'Inspection de Gradient 2 "
        "(désactivé) : « Gradient n'est pas calculé pour ce workflow. » alors que son bouton dit « À jour ».",
        "12_inspect_gradient2.png", ["Placer Gradient deux fois", "Désactiver Gradient 2", "Double-clic sur Gradient 2"]),
    rec("mineur", "Bandeau de phase « TRAITEMENT LINÉAIRE » coupé ; raisons de la palette hors de la page à 1366",
        "Avec une seule colonne d'image (projet Image + Gradient), le titre « TRAITEMENT LINÉAIRE » (127 px) dépasse sa "
        "bande (110 px) et est coupé (« TRAITEMENT LINÉAI ») ; en anglais « LINEAR PROCESSING ». À 1366×768 avec le "
        "Journal ouvert, la palette défile : survoler Split, Fusion ou Combiner (grisés) affiche leur raison sous le "
        "bas de la page (y 473–641 pour une page de 468 px), donc invisible.",
        "09_bande_coupee_journal_ouvert.png",
        ["1366×768, projet Image, placer Gradient", "Affichage > Journal", "Survoler « Combiner les calques… » dans la palette"]),
    rec("mineur", "Nouveau projet : le message « Absente des catalogues ici… » est coupé en bas",
        "Avec une cible inconnue, le message sur deux lignes n'a que 24 px de haut pour 32 : la deuxième ligne "
        "(« résolu à partir de ses étoiles seules ») est coupée à mi-hauteur. Aux deux tailles, dans les deux langues.",
        "10_nouveau_projet_message_coupe.png", ["Accueil > Nouveau projet…", "Cible : « Truc inconnu 42 », Tab"]),
    rec("mineur", "Ponctuation et accords en français",
        "Fenêtre d'exécution, infobulles des lignes : « Alignement: ≤ 1 min », « Astrométrie: ≈ 2 min », « Relevé "
        "H-alpha: … », « Inventaire: Non exécuté » (pas d'espace avant « : », texte assemblé dans le code). Combo du "
        "GPU : « cuda:0 · 24 GB » (identifiant interne, « GB » au lieu de « Go »). Message « Calque « Empilement brut » "
        ": sortie de Empilement » (élision). Journal : « Couleur ajouté », « Sélection ajouté ». « Gradient désactivé "
        "dans Full » (sans guillemets, ailleurs « Full » entre « »). Double « : » : « Astrométrie ne peut pas être "
        "retiré de « Full » : Couleur en a besoin : retirez d'abord Couleur ». Paramètres (mode spécification) : "
        "« Threads CPU ».",
        "11_fenetre_execution.png", ["Chaîne complète, Exécuter > Tout exécuter, survoler les lignes"]),
    rec("mineur", "Galerie : plus haute que l'écran 1366×768 ; noms coupés dans les aperçus",
        "La fenêtre des modèles fait 1180×700 : avec la barre de titre (~31 px) et la barre des tâches (~40 px), il "
        "reste 697 px sur un écran 768 : le bas (Annuler / Créer) touche ou passe sous la barre des tâches. Dans les "
        "aperçus, les noms des blocs sont coupés : « Empile… », « Débruit… », « Fusio… », « Combi… ».",
        "14_galerie_apercu.png", ["Nouveau projet à 1366×768"]),
    rec("question", "Noms de logiciels externes dans l'interface",
        "Infobulle « Balance des blancs de l'appareil » (panneau Empilement) : « l'option « Use camera white balance » "
        "de DeepSkyStacker » (et en anglais). À propos > Sources des données : « Siril Astrometry Catalogue … Siril » "
        "(citation du catalogue réellement lu). Paramètres > Catalogues (mode spécification ◇) : « ASTAP », "
        "« astrometry.net (local) ». Infobulle de la résolution à l'aveugle : « comme astrometry.net ».",
        "15_panneau_empilement.png", ["Projet mono, cliquer Empilement, survoler « Balance des blancs de l'appareil »"]),
    rec("question", "Mots internes et de banc d'essai dans des textes",
        "Lus dans le catalogue (pas tous vus à l'écran) : une infobulle de réglage dit « Désactivé par défaut : sur la "
        "Rosette, l'estimation contenait… Neutre sur Heart » (les projets de l'utilisateur) ; « Le programme est en "
        "mode hors ligne (ASTRO_GPU_OFFLINE) » ; « « none » : pas de relevé » (valeur d'option) ; filtre du dialogue "
        "Ouvrir « Projets créés avec une ancienne version (project.json) » ; « n'est pas une pose que le moteur lit » "
        "(panneau Session). Code : des erreurs Python sont montrées brutes (« Cette image ne peut pas être affichée : "
        "TypeErreur: … » dans l'Inspection, 1re ligne de l'erreur dans le bandeau du Post, « L'export a échoué : … »).",
        "", ["static_catalogue.txt, section « internal word »"]),
    rec("question", "Catalogue : une même clé traduite deux fois différemment",
        "Le dernier fichier lu gagne : « frames » = « images » (base) / « de poses » (graph) ; « Strength » = « Force » "
        "/ « Intensité » ; « Running » = « En cours » / « En cours d'exécution » ; « Dark scaling » = « … des darks » / "
        "« … du dark » ; « flat darks » = « darks de flats » / « darks de flat ». Selon l'écran, le mot attendu n'est "
        "peut-être pas celui affiché.", "", ["static_catalogue.txt, section « key given twice »"]),
    rec("question", "Messages trompeurs : « tout est à jour » sans rien à calculer ; raisons de Désactiver",
        "Ligne d'état « 1 workflow · tout est à jour » (et infobulle de Exécuter « Rien à calculer : tout est à "
        "jour. ») pour un workflow vide avec des sessions (two_sessions, session_empty_wf). Menu de Débruitage : "
        "« Désactiver — Split étoiles et fond en a besoin : retirez d'abord Split étoiles et fond » (une raison de "
        "suppression pour un désactiver). Menu du bloc Sortie : « Désactiver — ce bloc n'est pas dans le workflow ».",
        "", ["Projet two_sessions : ligne d'état", "Projet Full : menu de Débruitage, menu de Sortie"]),
]

(OUT / "anomalies.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in NOTES) + "\n",
                                     encoding="utf-8")
print(len(NOTES), "anomalies")
