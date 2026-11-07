"""palette_matrix, round 3: the verdicts (recheck.jsonl) from the RESULT lines of explore3_palette_recheck/more/around,
and the new anomaly (via ex.note)."""
from __future__ import annotations

import json

from harness import Explorer

ex = Explorer("palette_matrix")
D = ex.out


def s(n):
    f = D / f"{n}.png"
    return str(f) if f.is_file() else None


R = [
 ("Premier double-clic sur « Combiner les calques étoiles et fond » : rien ne change à l'écran", "corrigé",
  "Workflow complet : le premier double-clic pose un bloc « Recombiner 2 » et le message « Calque « Full · Recombiner 2 » (Recombiner) vers Post · Ctrl+Z » le dit.", s("r3_combiner_1")),
 ("Deux blocs « Combiner les calques étoiles et fond » portent exactement le même nom", "changé",
  "Les copies posées s'appellent « Recombiner 2 », « Recombiner 3 », mais le premier reste « Recombiner » sans numéro et, « Recombiner 2 » retiré, il reste « Recombiner » et « Recombiner 3 » (les Gradient et les Fusion, eux, se renumérotent 1, 2).", s("r3_fr_recombiner_noms")),
 ("Calque de Post créé par « Combiner » nommé en anglais", "corrigé",
  "Post liste maintenant les calques avant tout calcul, en français : « Full · Recombiner 2 · pas encore calculé ».", s("r3_combiner_post")),
 ("À l'ouverture, aucun bloc choisi : panneau de droite vide, la palette vise pourtant un workflow", "corrigé",
  "Projet Image seule : à l'ouverture le bloc Image est choisi, son panneau est à droite et la palette dit « pour « Image » ».", s("r3_image_ouverture")),
 ("Pause vérif sur une copie (Gradient 2, Couleur 2, Débruitage 3) : Exécuter répond « Rien à exécuter »", "corrigé",
  "Pause sur Gradient 2 : la fenêtre d'exécution liste Image, Astrométrie, Gradient 1, Gradient 2 et s'arrête là ; de même sur Couleur 2 et sur Débruitage 2 d'un workflow complet.", s("r3_pause_a2_fond2")),
 ("Projet Image seule : à l'ouverture la palette grise tout (« poser Session/Image d'abord ») alors que l'Image est posée", "corrigé",
  "À l'ouverture comme après un clic dans le vide, la palette dit « pour « Image » », Astrométrie et Gradient sont disponibles et un double-clic sur Gradient le pose dans le workflow Image.", s("r3_image_clic_vide")),
 ("Une seconde Fusion posée ne se voit pas : un seul bloc Fusion dessiné pour deux", "corrigé",
  "Deux blocs « Fusion 1 » et « Fusion 2 » sont dessinés, chacun avec son panneau ; la seconde se supprime seule et la première redevient « Fusion ».", s("r3_fusion_deux")),
 ("Fusion posée sur des canaux sans Astrométrie : la carte n'est pas grisée, le bloc arrive en rouge et Exécuter se grise", "corrigé",
  "La carte Fusion est grisée avec « Astrométrie d'abord dans Ha, OIII » et le double-clic ne pose rien.", s("r3_fusion_sans_astro_2")),
 ("Poser une Fusion ajoute Débruitage et Étoiles/fond dans les workflows des canaux", "corrigé",
  "Le message de pose le dit : « Fusion « HOO » posée · Débruitage, Étoiles/fond ajoutés à Ha, OIII · Ctrl+Z ».", s("r3_fusion_pose")),
 ("La raison de la carte Étoiles/fond parle de « Sortie »", "corrigé",
  "La raison dit maintenant « Étoiles/fond : il faut l'Empilement avant ».", None),
 ("Raisons de la palette : accords et élisions fautifs en français", "corrigé",
  "« une seule Calibration », « une seule Sélection », « pas d'Alignement à partir d'un fichier image », « pas d'Astrométrie à partir d'une image étirée ».", s("r3_raisons_etiree")),
 ("Branche avec un second Gradient : « Gradient » et « Gradient 2 », jamais « Gradient 1 »", "toujours là",
  "La branche « Full · Gradient 2 » contient toujours « Gradient » puis « Gradient 2 », sans « Gradient 1 ».", s("r3_branche_gradient")),
 ("1366×768 : en glissant un bloc, la plupart des places sont hors de l'écran", "toujours là",
  "Fenêtre 1366×768, carte Gradient tenue : 4 places sur 10 sont visibles, Post est hors de l'écran, et tenir la carte au bord droit ne fait pas défiler le graphe.", s("r3_fr_drag_1366x768")),
 ("Plusieurs nuits : ajouter une Session détache les nuits déjà là du workflow", "corrigé",
  "La Session 3 rejoint Session 1 et Session 2 (« « Session 3 » rejoint « Plusieurs nuits » · Ctrl+Z ») ; deux Ctrl+Z reviennent exactement à l'état d'avant.", s("r3_nuits_dbl")),
 ("Message « pourquoi grisé » resté affiché après Ctrl+Y, sur une carte devenue disponible", "corrigé",
  "Après Ctrl+Y la carte Astrométrie redevient disponible et le message disparaît.", s("r3_hint_ctrl_y")),
 ("Après Ctrl+Z d'un bloc posé, aucun bloc choisi et panneau de droite vide", "toujours là",
  "Après Ctrl+Z de l'Empilement, aucun bloc n'est surligné, le panneau de droite est vide, et Suppr ou D ne font rien sans un mot.", s("r3_fr_sel_apres_ctrl_z")),
 ("Fusion retirée : les blocs qu'elle avait ajoutés aux canaux restent, et deux Recombiner apparaissent", "corrigé",
  "Fusion retirée (glissée sur la palette ou Supprimer) : les canaux reviennent exactement comme avant, même si un Gradient a été ajouté à Ha entre-temps (il reste).", s("r3_fusion_edit_retrait")),
 ("Messages de retrait : accords fautifs et phrases tronquées", "corrigé",
  "« Calibration supprimée », « Astrométrie gardée : Couleur en a besoin », « Empilement gardé : les blocs suivants ont besoin de ce qu'il donne (image) ».", s("r3_retraits")),
 ("Modèle SHO, avant d'ajouter les brutes : les trois filtres dessinés comme une seule chaîne, et trois blocs « Sortie »", "corrigé",
  "Dès la création, trois chaînes séparées (trois Alignement, trois Empilement…) et aucun bloc « Sortie ».", s("r3_sho_ouverture")),
 ("Session ajoutée dans un projet de filtres : elle rejoint un canal sans son filtre, parfois pas celui affiché", "corrigé",
  "La nouvelle Session rejoint le canal nommé dans l'en-tête avec son filtre (Ha, ou SII dans le modèle SHO) et le message le dit ; avec une Fusion choisie, elle part dans un workflow à part, message à l'appui.", s("r3_session_filtre_new_sho")),
 ("Messages après une pose : « + ≈ ≤ 1 min à exécuter », et la branche nommée par son point de départ", "corrigé",
  "Le message dit « + ≤ 1 min à exécuter » et la branche sous Empilement s'annonce « Branche « Full · Gradient » créée · Ctrl+Z ».", None),
 ("Le mot anglais « off » dans l'interface française", "corrigé",
  "« Gradient désactivé · Full · Ctrl+Z » et « Alignement déjà là, désactivé : réactiver (D) ».", s("r3_off")),
]
with (D / "recheck.jsonl").open("w", encoding="utf-8") as fh:
    for t, st, n, f in R:
        fh.write(json.dumps({"title": t, "status": st, "note": n, "shot": f}, ensure_ascii=False) + "\n")
print(len(R), {k: sum(1 for x in R if x[1] == k) for k in ("corrigé", "toujours là", "changé")})

ex.note("mineur", "Deux Fusions : leurs deux blocs Recombiner portent le même nom « Recombiner »",
        "Projet mono (Ha, OIII, Astrométrie dans chaque canal). Deux Fusions posées : elles s'appellent bien « Fusion 1 » "
        "et « Fusion 2 », mais leurs deux blocs Recombiner s'appellent tous deux « Recombiner », sans numéro. Vu deux fois. "
        "Attendu (PRINCIPLES : les copies sont numérotées, même nom partout) : « Recombiner 1 » et « Recombiner 2 ».",
        shot=s("r3_fusion_deux") or True,
        steps=["projet mono Ha + OIII, Astrométrie dans chaque canal", "double-clic carte Fusion", "double-clic carte Fusion"])
ex.close()
