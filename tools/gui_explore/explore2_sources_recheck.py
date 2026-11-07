"""sources_files, round 2: the recheck of the round-1 anomalies, one line each, from what the explore2_sources_* runs
replayed (their logs and screenshots in the round-2 folder)."""
import json
from pathlib import Path

R1 = Path("F:/astro_essais/A_REGARDER/exploration/sources_files/anomalies.jsonl")
OUT = Path("F:/astro_essais/A_REGARDER/exploration2/sources_files")
C, T, X = "corrigé", "toujours là", "changé"
# round-1 line number -> (status, note, screenshot of round 2)
R = {
    1: (T, "Le panneau d'une Session n'a toujours pas de bouton Supprimer, seule ou avec une autre session ; le menu du "
           "bloc l'a.", "F_two_sessions_panel.png"),
    2: (C, "L'infobulle du Filtre dit maintenant à quoi sert le filtre (workflows par filtre et Fusion).",
        "A_session_panel.png"),
    3: (C, "Une Session seule a son menu : Renommer session, Supprimer grisé avec « seule session », Ajouter session…",
        None),
    4: (X, "Le premier bloc s'appelle maintenant « Session 1 · brutes à ajouter » (en pointillé) ; mais le panneau "
           "d'un projet sans bloc choisi s'appelle encore « Images » avec « Importer dossier… ».",
        "002_modèle_à_partir_de_brutes_le_premier_bloc_s_appell.png"),
    5: (C, "Choisir image… propose FITS, TIFF, PNG, JPEG et Tous les fichiers.", "img_B_fits.png"),
    6: (C, "Un PNG choisi met l'Image en « Étirée ».", "img_B_PNG.png"),
    7: (C, "Un JPEG choisi met l'Image en « Étirée ».", "img_B_JPG.png"),
    8: (C, "Le JPEG n'est plus compté : « 15 / 16 utilisées », avec son avertissement.", "C_jpg_in_lights.png"),
    9: (C, "« Plusieurs nuits » crée deux Sessions, chacune avec sa Calibration, puis une chaîne commune.",
        "created_nights.png"),
    10: (C, "Un fichier absent est refusé avec un message ; l'image d'avant reste.", "img_B_missing_file.png"),
    11: (C, "Un fichier texte est refusé : « n'est pas une image que cette version sait lire ».", "img_B_text_file.png"),
    12: (C, "Un fichier ou un dossier déjà là : message « … est déjà dans la session ».", None),
    13: (C, "Le dossier garde « 15 fichiers » et le fichier ajouté seul n'est compté qu'une fois.",
         "D_file_of_folder_added_alone.png"),
    14: (X, "Un dossier vide ou sans poses n'est plus ajouté : message « ne contient aucune brute : non ajouté ».",
         "E_empty_folders.png"),
    15: (C, "« Image déjà étirée » crée Image → Post, l'image marquée Étirée, sans Astrométrie ni Couleur.",
         "created_image_stretched.png"),
    16: (X, "La galerie propose maintenant de choisir les filtres (L R G B Ha OIII SII) ; Créer s'active dès qu'on en "
            "choisit. Mais le projet créé dessine les filtres en une seule chaîne avec des blocs « Sortie » (nouvelle "
            "anomalie).", "gallery_mono_each.png"),
    17: (C, "Changer le filtre s'annule (Ctrl+Z) et se rétablit (Ctrl+Y).", "H_filter_Ha.png"),
    18: (C, "Plus aucun texte anglais dans la galerie en français.", None),
    19: (C, "Renommer une session (panneau ou menu) s'annule et se rétablit.", None),
    20: (C, "Les deux Sessions restent dessinées après « + Session » dans un projet sans workflow.",
         "I_two_sessions_no_workflow.png"),
    21: (T, "Deux blocs Image s'appellent toujours « Image » tous les deux (le nom du workflow n'est que dans le "
            "résumé).", None),
    22: (C, "La fenêtre d'import dit « Ni offsets ni darks de flat. »", None),
    23: (C, "La fenêtre d'import puis la galerie s'ouvrent l'une après l'autre, jamais ensemble.", None),
    24: (C, "Le projet est proposé dans le dossier des projets, pas dans celui des brutes.", None),
    25: (C, "Le projet importé est créé dans le dossier des projets.", None),
    26: (C, "Importer de nouveau le même dossier n'ouvre plus de fenêtre bloquante.", None),
    27: (C, "Le nom du canal et son texte sont séparés (« Ha · gris »).", "fus_HaOIII_panel.png"),
    28: (C, "Un dossier vide ne crée plus de projet ; un message le dit.", None),
    29: (C, "Un dossier de JPEG ne crée plus de projet ; un message dit qu'il n'y a pas de brutes lues.", None),
    30: (C, "Un projet abîmé : message en clair, sans erreur Python.", None),
    31: (C, "Créer grisé a une infobulle : « Aucun fichier image choisi pour l'instant. »",
         "T_gallery_image_empty.png"),
    32: (C, "Créer grisé a une infobulle : « Fichier introuvable : absente.fits ».", "T_gallery_image_missing.png"),
    33: (X, "En HOO, SII n'est plus dans les canaux réglables : une ligne dit « Non lus par HOO : SII ».",
         "fus_palette_HOO.png"),
    34: (C, "Le panneau Fusion a son bouton Supprimer.", "fus_SHO3_panel.png"),
    35: (C, "Supprimer la Session Ha d'un projet avec Fusion ne lève plus d'erreur.", "fus_session_removed.png"),
    36: (X, "La Fusion est entourée de rouge et Exécuter est grisé ; la Session vidée reste dessinée, mais sa ligne "
            "passe en bas et les fils se croisent (nouvelle anomalie).", "fus_session_removed.png"),
    37: (C, "Le filtre est repris de l'en-tête des FITS (« · SII » sur le bloc).", "M_SII_added.png"),
    38: (T, "Deux Fusions portent toujours le même titre « Fusion » (seul le résumé dit « SHO 2 »).",
         "fus_two_fusions.png"),
    39: (C, "La Fusion bi-bande placée depuis la palette prend les étoiles du workflow couleur, comme le modèle.",
         "fus_colour_panel.png"),
    40: (C, "Le message dit « Fusion et suivants retirés · Ctrl+Z » (et « Image et suivants retirés »).", None),
    41: (C, "La vérification des images signale le JPEG et le PNG des brutes.", None),
    42: (C, "La vérification des images signale les offsets CR2 parmi des brutes FITS.", None),
    43: (C, "La colonne Filtre d'Inspection montre le filtre de la session (OIII).", None),
    44: (C, "« + Session » et la zone « Flux » ne se recouvrent plus ; un clic au milieu ajoute la session.",
         "I_two_sessions.png"),
    45: (T, "Ctrl+Z après avoir retiré une brute défait encore la Calibration posée avant, sans prévenir.",
         "006_ctrl_z_après_avoir_retiré_un_fichier_le_fichier_re.png"),
    46: (C, "Calibration et zone « Flux » ne se chevauchent plus.", "I_session2_panel.png"),
    47: (C, "À 1366×768, Image et la zone « Flux » ne se chevauchent plus.", "small_fr_1366_image_panel.png"),
    48: (C, "Les menus en français affichent « Suppr ».", None),
    49: (C, "Un seul nom partout : « Darks de flat » en français, « Dark flats » en anglais.", None),
    50: (C, "Message clair : « … n'est pas un fichier de modèle de ce programme. »", None),
    51: (C, "La barre d'état dit « 1 workflow · aucun bloc : en poser un ».", "small_fr_1366_session_panel.png"),
    52: (C, "Le bloc s'appelle « Étoiles/fond ».", None),
}
r1 = [json.loads(x) for x in R1.read_text(encoding="utf-8").splitlines() if x.strip()]
assert len(r1) == len(R), (len(r1), len(R))
with (OUT / "recheck.jsonl").open("w", encoding="utf-8") as fh:
    for i, a in enumerate(r1, 1):
        st, note, shot = R[i]
        f = OUT / shot if shot else None
        fh.write(json.dumps({"title": a["title"], "status": st, "note": note,
                             "shot": str(f) if f is not None and f.is_file() else None}, ensure_ascii=False) + "\n")
counts = {}
for st, _n, _s in R.values():
    counts[st] = counts.get(st, 0) + 1
print(counts)
