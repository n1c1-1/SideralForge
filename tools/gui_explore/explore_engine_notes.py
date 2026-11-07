"""Engine exploration (engine_shapes): the anomalies found, each reproduced at least twice, written for the user to
judge (F:/astro_essais/A_REGARDER/exploration2/engine_shapes/anomalies.jsonl). Run once."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_engine_common import note  # noqa: E402

PY = "python tools/gui_explore/"
NOTES = [
    ("bloquant",
     "Les réglages d'un bloc placé deux fois (Gradient 2, Denoise 2, Color 2) ne peuvent pas être changés",
     "Le panneau d'un bloc répété liste ses réglages sous l'étape « gradient~2 » (settings._stages_of_step). Changer "
     "un réglage passe par model.throws_away puis model.apply_change → api.set_param(…, 'gradient~2', …) → "
     "_suggestion → KeyRules.defaults('gradient~2') → KeyError 'gradient~2' (la table des étapes ne connaît que "
     "« gradient »). Le panneau n'attrape que ValueError : l'exception sort dans le slot Qt, rien n'est enregistré. "
     "Testé pour degree, rbf_smooth, samples de Gradient 2 ; le même réglage de Gradient 1 passe. api.describe "
     "de « gradient~2 » fonctionne (le panneau s'affiche), seul le changement échoue.",
     ["image linéaire → [Gradient polynomial, Gradient 2 rbf]",
      "model.apply_change([vue], {'kind':'param','stage':'gradient~2','name':'degree','value':4})",
      PY + "explore_engine_repro.py panel_setting_of_gradient2"]),
    ("gênant",
     "Fenêtre d'exécution : le Gradient 2 reste « non lancé » après un calcul réussi, ses événements vont au Gradient 1",
     "Les événements du job d'un bloc répété (stage « gradient~2 ») sont émis par le code de l'étape sous le nom "
     "« gradient » : RunJournal.emit n'y ajoute ni key, ni wf, ni block (il compare stage == job.stage). Le suivi du "
     "client (services/run.RunTrack._job) les rattache alors par nom d'étape au job « gradient » : le Gradient 1 "
     "repasse « en cours », et le Gradient 2 ne reçoit jamais son début ni sa fin. Après un calcul « done », RunTrack "
     "donne : image done, gradient done, gradient~2 not_started, render_simple done. Un échec du Gradient 2 serait "
     "affiché sur le Gradient 1. Vérifié aussi pour Denoise 2 (poses 860 px : son début et sa fin sans key ni block) ; même cause pour Color 2.",
     ["image linéaire → [Gradient, Gradient 2 rbf], Run (CPU)",
      "events.jsonl : 4 événements stage_start/progress/stage_end « gradient » sans key",
      PY + "explore_engine_repro.py run_window_of_repeated_blocks"]),
    ("gênant",
     "Un bloc éteint puis le même bloc reposé depuis la palette : la palette propose des places, le dépôt échoue "
     "avec un message interne",
     "Calibration, Registration, Selection ou Astrometry éteint (D) : les règles comptent un bloc éteint comme absent, "
     "donc la palette ne grise pas le bloc et propose des places. Au dépôt, le nouveau nœud est nommé "
     "« registration~2 » et set_blocks refuse : le toast affiche « block 'registration' cannot be named "
     "'registration~2' » (anglais, nom interne, même en français). Trouvé 20 fois par le balayage (règles et modèle) "
     "et reproduit pour les quatre blocs. Attendu : soit le bloc grisé avec « un workflow a un seul bloc "
     "Registration », soit le dépôt accepté.",
     ["workflow de poses [Calibration, Registration, Selection, Stacking, Astrometry, Gradient]",
      "éteindre Registration, puis glisser Registration depuis la palette à une place allumée",
      PY + "explore_engine_repro.py second_block_after_switch_off"]),
    ("question",
     "Un bloc reposé après suppression reprend les réglages du bloc supprimé ; une branche reprend ceux d'un autre bloc",
     "Les réglages d'un bloc répété sont rangés sous son nom (« gradient~2 ») et ne sont pas effacés quand le bloc "
     "est supprimé. Reposer un Gradient lui redonne le nom « gradient~2 » : il revient avec les réglages de "
     "l'ancien (ex. degree 4), sans que l'utilisateur les ait mis. De même, une branche copie tous les réglages du "
     "tronc : le nouveau Gradient de la branche, nommé « gradient~2 », prend les réglages du Gradient 2 du tronc, qui "
     "est un autre bloc placé ailleurs. Voulu ?",
     ["image → [Gradient, Astrometry, Gradient 2 (degree 4)] ; supprimer Gradient 2 ; reposer un Gradient",
      "tronc [Gradient, Astrometry, Gradient 2 (degree 5)] ; branche après Gradient 1 avec un Gradient",
      PY + "explore_engine_repro.py settings_of_repeated_blocks"]),
    ("mineur",
     "Le même calcul a une autre clé selon le nom interne des blocs : Sortie recalculée en éteignant un bloc, "
     "flux identiques non partagés",
     "La clé d'une étape contient le nom de ce qu'elle lit (« gradient » ou « gradient~2 »), pas seulement sa clé. "
     "Éteindre le Gradient 1 de [Gradient poly, Gradient 2 rbf] réutilise bien le résultat du Gradient 2, mais la "
     "Sortie change de clé par rapport au même graphe sans le bloc éteint : elle est recalculée alors que l'image est "
     "la même. Deux flux [rbf puis polynomial] construits dans un ordre de dépôt différent (le second bloc posé en "
     "premier) donnent des clés différentes pour le second Gradient et la Sortie : calculés deux fois au lieu d'être "
     "partagés. Coût faible sur une image, réel sur un Denoise placé après.",
     [PY + "explore_engine_image.py switched_off", PY + "explore_engine_image.py keys_properties"]),
    ("question",
     "Deux images différentes de même nom, taille et date (dossiers différents) partagent un seul résultat",
     "La clé d'une source image est [nom, taille, date à 2 s près], sans le dossier (pour qu'une copie garde ses "
     "résultats). Deux fichiers « stack.fits » de même taille (même dimensions) écrits dans la même seconde par un "
     "export par lot : le second flux montre l'image et les résultats du premier (vérifié : le drz_full du flux B est "
     "l'image A).",
     [PY + "explore_engine_image.py two_files_same_name"]),
    ("gênant",
     "Une Fusion dont un canal est un bloc Image sans fichier est lancée quand même et échoue",
     "Le planificateur écarte bien le workflow Image sans fichier (« its Image block has no file yet ») mais la "
     "Fusion qui le lit est planifiée, et elle replanifie l'image de ce canal : Run all calcule les autres canaux, "
     "puis échoue sur « no image file: choose the image this workflow starts from ». Le plan est incohérent "
     "(skipped {2: …} et des jobs du workflow 2). Attendu : la Fusion écartée avec la raison, rien de lancé pour elle.",
     ["image A [Gradient] ; image B posée sans fichier ; Fusion HOO (A → Ha, B → OIII)",
      "runner2.plan_all → skipped {B}, jobs image/render_simple de B, fusion, render_simple",
      PY + "explore_engine_fusion.py input_without_file"]),
    ("mineur",
     "Denoise échoue sur une pile de moins de 800 px de côté (erreur Python brute)",
     "L'aperçu avant/après de Denoise découpe un carré de 800 px au centre et le concatène à une bande de 800 px de "
     "haut : sur une pile plus petite (frames 200×300 ou 560×560 du banc), l'étape échoue après avoir calculé, avec "
     "« ValueError: all the input array dimensions except for the concatenation axis must match exactly… » dans la "
     "fenêtre d'exécution ; Split stars derrière est sauté. Rare sur de vraies piles, mais l'erreur est brute et "
     "empêche les essais rapides sur petites images. À 860 px tout passe.",
     ["poses Bayer 200×300 (8) ou 560×560 (6) → [Calibration, Registration, Stacking, Gradient, Denoise], Run CPU",
      PY + "explore_engine_frames.py denoise_cpu"]),
    ("mineur",
     "Dossier de poses introuvable (disque débranché) : le message dit d'ajouter des poses",
     "Une session dont le dossier n'existe plus est planifiée normalement ; l'Inventaire échoue sur « no light frame "
     "found: add at least one folder of light frames », alors que le dossier est bien là dans la session, seulement "
     "inaccessible. Attendu : dire que le dossier (ou le disque) est introuvable. Même message pour une session vide.",
     ["session 1 → lights = [un dossier qui n'existe pas] ; [Stacking] ; Run",
      PY + "explore_engine_frames.py missing_folder"]),
    ("mineur",
     "Image linéaire : Split stars grisé « il faut Denoise avant », mais Denoise est impossible sur une image",
     "Sur un workflow Image linéaire, la palette grise Split stars avec « it needs the Denoise block before it » ; "
     "Denoise y est grisé à son tour (« it needs the half-stacks of a stack »). La première raison envoie "
     "l'utilisateur vers un bloc qu'il ne peut pas poser ; dire directement que Split stars n'est pas possible sur "
     "une image linéaire (seulement sur une image déjà étirée ou un workflow de poses).",
     ["workflow Image linéaire [Gradient] ; palette : Split stars, puis Denoise",
      PY + "explore_engine_repro.py stars_reason_on_image"]),
    ("question",
     "Fusions qui ne peuvent pas aboutir : refusées seulement au calcul",
     "Deux règles de la Fusion ne sont vérifiées qu'à l'exécution, après le calcul des canaux : deux workflows Image "
     "sans Astrometry (« the channels come from different stacks: each needs its Astrometry… ») et une Fusion avec "
     "étoiles dont les canaux n'ont pas Split stars (« the Ha channel has no starless image… », impossible pour un "
     "canal Image linéaire, qui n'a pas de Denoise). Les messages sont clairs, mais le principe « tout graphe permis "
     "tourne ou est refusé avec sa raison » suggérerait de les dire avant Run.",
     [PY + "explore_engine_fusion.py two_images_no_astrometry", PY + "explore_engine_fusion.py fusion_with_stars_plan"]),
]

if __name__ == "__main__":
    for sev, title, detail, steps in NOTES:
        note(sev, title, detail, steps)
    print(len(NOTES), "notes written")
