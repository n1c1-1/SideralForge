"""sources_files, round 3: the recheck of the round-2 open points (recheck lines not fixed, and the round-2 anomalies),
one line each, from what the explore3_sources_* runs replayed (logs in exploration3/sources_files/logs); the new
problems noted with ex.note."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Explorer  # noqa: E402

R2 = Path("F:/astro_essais/A_REGARDER/exploration2/sources_files")
OUT = Path("F:/astro_essais/A_REGARDER/exploration3/sources_files")
C, T, X = "corrigé", "toujours là", "changé"
# ("recheck" | "anomalies", line number in the round-2 file) -> (status, note, screenshot of round 3)
R = {
    ("recheck", 0): (C, "Le panneau d'une Session a maintenant son bouton Supprimer : grisé avec « Suppression impossible : "
                        "seule session » quand elle est seule, actif à deux sessions (« Session 2 » supprimée, Ctrl+Z la "
                        "rend).", "F_two_sessions_panel.png"),
    ("recheck", 3): (C, "Le projet créé s'ouvre sur « Session 1 · brutes à ajouter » ; un clic dans le vide laisse le "
                        "panneau vide, il ne montre plus « Images / Importer dossier… ».", "x_tpl_full_nothing_selected.png"),
    ("recheck", 13): (C, "Un dossier vide ou sans poses est refusé avec sa raison : « « vide » ne contient aucune brute "
                         "(RAW d'appareil ou FITS) : non ajouté. »", "x_empty_folder_Flats.png"),
    ("recheck", 15): (C, "Modèle « Un workflow par filtre » (Ha, OIII) : deux chaînes séparées dès la création, plus "
                         "aucun bloc « Sortie ».", "J_created_mono_each_2.png"),
    ("recheck", 20): (T, "Une seconde Image déposée sur la zone du bas : les deux blocs s'appellent toujours « Image » "
                         "(seul le résumé dit « Image 2 · fichier à choisir »).", "x_two_images.png"),
    ("recheck", 32): (C, "En HOO, SII n'a plus de poids réglable : une ligne dit seulement « Non lus par HOO : SII ».",
                      "fus_palette_HOO.png"),
    ("recheck", 35): (C, "Supprimer la Session Ha d'un canal est refusé avec la raison : « Session « Ha » gardée : canal "
                         "Ha de « SHO » » ; la Fusion reste entière.", "fus_session_removed.png"),
    ("recheck", 37): (C, "Deux Fusions s'appellent maintenant « Fusion 1 » et « Fusion 2 » sur le graphe.",
                      "fus_two_fusions.png"),
    ("recheck", 44): (T, "Ctrl+Z juste après avoir retiré une brute (× de la liste) défait encore les deux Calibrations "
                         "posées avant (2 → 0) et la brute reste retirée, sans prévenir.", "I_file_out_then_undo.png"),
    ("anomalies", 0): (C, "Alignement du workflow OIII choisi, Ctrl+S sans autre changement, fermer, rouvrir : le projet "
                          "revient sur cet Alignement.", "A_reopen_selection_only.png"),
    ("anomalies", 1): (C, "L'Image choisie, puis la Fusion choisie (deux fois) : à la réouverture le projet revient sur "
                          "elles, avec leur panneau.", "x3_fusion_reopened_1.png"),
    ("anomalies", 2): (C, "Une pile FITS choisie après un PNG remet l'Image en « Linéaire » (le PNG l'avait mise en "
                          "« Étirée »).", None),
    ("anomalies", 3): (C, "Fusion posée puis supprimée (croix, menu ou Suppr) : le projet revient exactement à l'état "
                          "d'avant, sans Débruitage, Étoiles/fond ni Recombiner en trop.", "G_fusion_deleted.png"),
    ("anomalies", 4): (C, "La nuit ajoutée par « + Session » rejoint le workflow : Calibration posée ensuite une fois "
                          "par session (deux blocs).", "H_calibration_after_plus_session.png"),
    ("anomalies", 5): (C, "Supprimer la Session d'un canal de la Fusion est refusé avec la raison : les lignes restent en "
                          "place, aucun fil croisé.", "I_session_removed_2.png"),
    ("anomalies", 6): (C, "Juste créé (Ha, OIII), le modèle dessine deux chaînes séparées, sans bloc « Sortie ».",
                       "J_created_mono_each_2.png"),
    ("anomalies", 7): (C, "La Session déposée sur « + Session » (sous la Session 2) rejoint « Flow » et le message le "
                          "dit : « « Session 3 » rejoint « Flow » · Ctrl+Z ».", "K_session_dropped_sion.png"),
    ("anomalies", 8): (C, "Les raisons disent « pas d'Alignement », « pas d'Empilement à partir d'un fichier image ».",
                       None),
    ("anomalies", 9): (C, "Après Créer avec « Complet » ou « Plusieurs nuits », c'est la Session 1 qui est choisie, son "
                          "panneau à droite.", "x_tpl_full_created.png"),
    ("anomalies", 10): (C, "Le résumé dit « 6 brutes · 3 darks · 3 flats » (pluriel partout).", "E_files_picked.png"),
    ("anomalies", 11): (C, "Modèle « Image déjà étirée » avec un PNG : Exécuter est actif (« ≈ 1 min »), la barre "
                           "d'état dit « à exécuter » et Post annonce « 1 calque ».", "x3_stretched_template_1.png"),
    ("anomalies", 12): (C, "Suppr ou glisser la Session seule vers la palette : refusé avec un message « Dernière "
                           "session : gardée ».", "S_lone_session_dragged.png"),
    ("anomalies", 13): (C, "Les projets créés depuis un modèle, sans brutes, disent « Pas encore d'images » sur "
                           "l'Accueil.", "V_home_cards.png"),
    ("anomalies", 14): (C, "Le dossier de FITS Ha importé donne la session « 2026-11-07 · Ha », filtre Ha.",
                        "X_after_import_ha.png"),
    ("anomalies", 15): (C, "Les FITS OIII importés ensuite (Ctrl+I) ont leur session « … · OIII » et leur propre "
                           "workflow « OIII », séparé de Ha.", "X_after_import_oiii.png"),
    ("anomalies", 16): (C, "Dossier « NGC7000 2026-08-01/Ha » : nom proposé « NGC7000 2026-08-01 », cible « NGC7000 ».",
                        "X_import_dialog_1.png"),
    ("anomalies", 17): (C, "Fichier disparu : le bloc Image est entouré de rouge, Exécuter est grisé et la barre d'état "
                           "dit « exécution refusée ».", "Y_image_file_gone.png"),
    ("anomalies", 18): (C, "Modèle puis deux Ctrl+I : la Session vide du modèle reçoit le premier dossier, le second "
                           "devient « 2026-11-07 (2) » dans le même workflow ; Ctrl+Z / Ctrl+Y suivent.",
                        "Z_after_ctrl_i_undo_redo.png"),
}
lines = {k: [json.loads(x) for x in (R2 / f"{k}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
         for k in ("recheck", "anomalies")}
counts = {}
with (OUT / "recheck.jsonl").open("w", encoding="utf-8") as fh:
    for (k, i), (st, note, shot) in R.items():
        f = OUT / shot if shot else None
        fh.write(json.dumps({"title": lines[k][i]["title"], "status": st, "note": note,
                             "shot": str(f) if f is not None and f.is_file() else None}, ensure_ascii=False) + "\n")
        counts[st] = counts.get(st, 0) + 1
print(counts)

if "--notes" in sys.argv:
    ex = Explorer("sources_files", lang="fr")
    ex.note("gênant", "Projet créé depuis un modèle : le premier Ctrl+Z vide la page",
            "Nouveau projet, galerie « Empilement classique » (puis « Complet »), Créer : la chaîne est dessinée. Un "
            "seul Ctrl+Z retire tout le workflow (blocs : alignement … sources → aucun), la barre d'état dit « Annulé : "
            "Workflow « Complet » créé » ; Ctrl+Y le remet. Vu deux fois (deux modèles). Attendu (décision 6 du tour 2) : "
            "la création depuis un modèle n'est pas une étape d'annulation, le premier Ctrl+Z ne vide pas la page.",
            shot=str(OUT / "x3_template_ctrl_z_2.png"),
            steps=["Nouveau projet", "galerie : « Complet » (ou « Empilement classique »)", "Créer", "Ctrl+Z"])
    ex.note("question", "Un TIFF 8 bits (photo déjà étirée) choisi reste « Linéaire », un PNG ou un JPEG passe en « Étirée »",
            "Image posée, Changer image… > un TIFF 8 bits RVB : la nature reste « Linéaire » (vu deux fois), alors qu'un "
            "PNG ou un JPEG de la même photo met l'Image en « Étirée ». Un TIFF 8 bits ne peut pas sortir tel quel d'un "
            "empilement linéaire ; les TIFF 16 et 32 bits restent bien « Linéaire ». À trancher : un TIFF 8 bits "
            "compté comme une photo étirée ?",
            shot=str(OUT / "x3_tiff8_1.png"),
            steps=["projet Image", "clic sur l'Image", "Changer image… > un TIFF 8 bits"])
    ex.close()
