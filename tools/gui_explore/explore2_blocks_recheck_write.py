"""block_actions round 2: recheck.jsonl written from the replays (explore2_blocks_recheck.py, explore2_blocks_dig.py),
each status and sentence checked by hand against the screenshots."""
import json
from pathlib import Path

D = Path("F:/astro_essais/A_REGARDER/exploration2/block_actions")
R1 = [json.loads(x) for x in Path("F:/astro_essais/A_REGARDER/exploration/block_actions/anomalies.jsonl")
      .read_text(encoding="utf-8").splitlines() if x.strip()]
SHOTS = {i + 1: json.loads(x)["shot"] for i, x in enumerate((D / "recheck_raw.jsonl").read_text(encoding="utf-8")
                                                            .splitlines()[:28])}
SHOTS[1] = str(D / "dig_fusion_session_cross.png")
SHOTS[25] = str(D / "dig_sortie_delete.png")

V = {
    1: ("changé", "Plus d'erreur Python : la croix, Suppr et le menu retirent la Session Ha (message « « Ha » retirée · "
        "fichiers gardés · Ctrl+Z »), Ctrl+Z, Ctrl+Y et Ctrl+S marchent ; mais à sa place reste une Session « Ha » vide "
        "(« brutes à ajouter ») et la Fusion est entourée de rouge au lieu d'un refus avec la raison ; par « Supprimer "
        "workflow » du dernier bloc de Ha, Ctrl+Y lève toujours l'erreur Python (noté à nouveau)."),
    2: ("corrigé", "« Pause vérif » sur Gradient 1 pose la pause sans erreur, message « Pause vérif après Gradient 1 »."),
    3: ("corrigé", "Gradient 1 s'appelle Gradient 1 dans le graphe, le panneau, le message de D et le calque envoyé "
        "(« Sortie Gradient 1 »)."),
    4: ("corrigé", "« Sans étoiles vers Post » ajoute 2 calques, Ctrl+Z les retire, Ctrl+Y les remet."),
    5: ("toujours là", "Après « Vers Post » sur Gradient (menu ou port tiré sur Post), un bloc « Sortie » est encore "
        "dessiné entre Couleur et Post."),
    6: ("toujours là", "Le menu de Calibration propose toujours « Retirer de Sortie Gradient » et « Désactiver dans "
        "Sortie Gradient » (les mots ont seulement été raccourcis)."),
    7: ("changé", "La copie est maintenant un vrai graphe, mais Dupliquer ne dit toujours rien (seul le coût s'affiche) "
        "et deux blocs « Sortie » apparaissent à droite."),
    8: ("corrigé", "« Supprimer workflow » agit sans fenêtre, avec « Workflow « Workflow » supprimé · Ctrl+Z », et "
        "Ctrl+Z le rend."),
    9: ("corrigé", "Suppr sur Gradient affiche « Gradient retiré · Ctrl+Z »."),
    10: ("corrigé", "Le menu écrit maintenant « Supprimer  Suppr »."),
    11: ("changé", "Les anciennes phrases ont disparu, mais les nouveaux messages gardent le masculin pour des noms "
         "féminins (« Calibration retiré », « Astrométrie retiré », « Astrométrie gardé ») et D affiche le mot anglais "
         "« off » (« Calibration off · Workflow »)."),
    12: ("corrigé", "Le menu dit partout « Vers Post », « Sans étoiles vers Post », « Étoiles vers Post »."),
    13: ("corrigé", "Le clic droit sur Gradient le sélectionne, le panneau montre Gradient."),
    14: ("changé", "La Session seule a maintenant son menu (Renommer, Supprimer grisé avec « seule session », Ajouter), "
         "mais le clic droit sur Post n'ouvre toujours rien."),
    15: ("corrigé", "« Ouvrir Post » grisé a une infobulle : « Aucun calque n'est encore calculé : exécutez d'abord un "
         "workflow »."),
    16: ("corrigé", "Plus aucun chevauchement de la zone « Flux » avec les blocs dans les trois cas rejoués."),
    17: ("corrigé", "Après un clic sur Empilement le clavier répond : D et Suppr donnent la raison du refus."),
    18: ("corrigé", "Renommer une session s'annule par Ctrl+Z et revient par Ctrl+Y (toujours sans message après le "
         "renommage)."),
    19: ("changé", "Le menu de la Calibration d'une session a maintenant Supprimer et Désactiver, qui marchent, mais son "
         "panneau n'a toujours ni bouton Supprimer ni Désactiver (seulement « Réglages avancés »)."),
    20: ("changé", "Le menu de la Fusion a gagné « Vers Post », mais toujours ni Inspection ni Pause vérif."),
    21: ("toujours là", "D sur le bloc Image ne fait rien et ne dit rien ; son menu n'a que « Supprimer »."),
    22: ("corrigé", "Le panneau montre « Désactiver » grisé avec l'infobulle « requis par Couleur : désactiver Couleur "
         "avant »."),
    23: ("corrigé", "Le bloc s'appelle « Étoiles/fond » et les menus en français n'ont plus « Split »."),
    24: ("corrigé", "La croix d'une Astrométrie partagée par une branche Couleur est refusée en entier, avec "
         "« Astrométrie gardé : requis par Couleur » (accord à revoir)."),
    25: ("toujours là", "Le bloc « Sortie » répond encore à Supprimer par « Sortie Gradient gardé : absent du workflow » ; "
         "sa croix et Suppr ne font rien ; son menu dit « Désactiver — absent du workflow »."),
    26: ("changé", "L'Astrométrie du canal Ha est toujours supprimée sans refus (message « Astrométrie retiré · Ctrl+Z ») ; "
         "la Fusion est maintenant entourée de rouge et Exécuter est grisé."),
    27: ("corrigé", "Avec une branche Couleur, « Désactiver » d'Astrométrie est grisé avec la raison, plus proposé puis "
         "refusé."),
    28: ("corrigé", "La croix de Recombiner le retire avec « Recombiner retiré · Ctrl+Z », Ctrl+Z le rend, et un simple "
         "clic ne crée plus de calque."),
}
with (D / "recheck.jsonl").open("w", encoding="utf-8") as fh:
    for k in range(1, 29):
        st, note = V[k]
        fh.write(json.dumps({"title": R1[k - 1]["title"], "status": st, "note": note, "shot": SHOTS.get(k)},
                            ensure_ascii=False) + "\n")
print({s: sum(1 for v in V.values() if v[0] == s) for s in ("corrigé", "toujours là", "changé")})
