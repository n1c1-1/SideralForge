"""Round 2 fuzz: writes recheck.jsonl from the replays of explore2_fuzz_recheck.py (judged by hand, 2 replays each)."""
import json
from pathlib import Path

OUT = Path("F:/astro_essais/A_REGARDER/exploration2/fuzz")
R1 = [json.loads(x) for x in Path("F:/astro_essais/A_REGARDER/exploration/fuzz/anomalies.jsonl").read_text(
    encoding="utf-8").splitlines() if x.strip()]
J = {
    1: ("changé", "Plus d'erreur ni de projet cassé : glisser « Ha » sur la palette (ou Suppr, ou le menu) remplace la "
               "session par une session « Ha » vide, redescendue sous OIII, et la Fusion est cerclée de rouge ; mais le "
               "message dit « Ha retirée » alors que le bloc Ha reste à l'écran (voir la nouvelle note)."),
    2: ("corrigé", "Le modèle « depuis les poses » crée maintenant une Session 1 dès le départ ; une Session glissée sur "
                   "la zone du bas ajoute Session 2 et son flux, sans erreur, projet valide."),
    3: ("corrigé", "Clic droit sur le fil Image → Post puis « Retirer » : le fil et le calque partent, sans erreur, et "
                   "Ctrl+Z le rend."),
    4: ("corrigé", "Retirer le fil vers Post (clic + Suppr ou menu) et « Vers Post » depuis le menu d'un bloc créent une "
                   "entrée d'annulation ; Ctrl+Z puis Ctrl+Y rendent exactement le Post d'avant et d'après."),
    5: ("corrigé", "Une Session glissée sur la zone du bas d'un projet SHO ajoute « Session 4 », dessinée, avec son "
                   "flux ; un double-clic sur Session dans un projet RVB ajoute une Session dessinée ; lâchée sur un "
                   "endroit vide, la carte est refusée (rien ne change)."),
    6: ("corrigé", "Avec deux flux, puis avec Ha, OIII et une Fusion : une session ajoutée puis retirée ne laisse aucun "
                   "fil dans le vide."),
    7: ("corrigé", "Poser une Session (zone du bas) ou une Image (double-clic) ne crée plus qu'une entrée "
                   "d'annulation : un seul Ctrl+Z revient à l'état d'avant."),
    8: ("corrigé", "Le nom changé dans le panneau de la Session s'annule par un Ctrl+Z (retour à « Session 1 »)."),
    9: ("corrigé", "L'infobulle « Jusqu'ici » d'Inspection dit maintenant « Calibration · Session 1, Calibration · "
                   "Session 2, Alignement, Empilement, Gradient 1, Gradient 2 », sans clé interne."),
    10: ("corrigé", "Couleur posée deux fois dans le même workflow (Color 1, Color 2) : « Pause vérif » sur l'un et "
                    "l'autre et un réglage changé dans le panneau de Color 2 marchent, sans erreur."),
    11: ("corrigé", "Deux flux, Alignement, Empilement glissé dans « Flow », puis Calibration : plus d'erreur du "
                    "panneau."),
    12: ("corrigé", "Ctrl+Z ou Suppr qui fait partir le bloc affiché (Image 2, Session 2) et la séquence complète du "
                    "premier tour : plus d'erreur du panneau quand il se reconstruit ; reste une erreur voisine, d'un "
                    "champ du panneau qui écrit dans le bloc parti (voir la nouvelle note)."),
    13: ("corrigé", "« Supprimer workflow » ne pose plus de question : le workflow part tout de suite, message court et "
                    "Ctrl+Z ; l'article a disparu mais le menu d'Empilement garde « Supprimer workflow » à côté de "
                    "« Supprimer »."),
    14: ("corrigé", "Session posée deux fois sur un projet d'image, deux Images et deux Sélections sur le projet au "
                    "JPEG : plus aucun bloc superposé (mais deux Images sans fichier ne font qu'un bloc : nouvelle "
                    "note)."),
    15: ("corrigé", "« Ajouter une session… » sur G crée une entrée d'annulation (Ctrl+Z la retire) ; reste vrai : le "
                    "dossier FITS dont l'en-tête dit FILTER = OIII ou SII donne une session sans filtre, nommée "
                    "d'après le dossier."),
    16: ("changé", "Supprimer la Fusion d'un projet HOO retire maintenant « Fusion et suivants » (Recombiner aussi), "
                   "avec Ctrl+Z ; plus de blocs Combiner en double ni de graphe en retard."),
    17: ("corrigé", "Le modèle « Image déjà étirée » crée une Image étirée qui va droit au Post (Image → Post), sans "
                    "Astrométrie, Gradient ni Couleur."),
}
with (OUT / "recheck.jsonl").open("w", encoding="utf-8") as fh:
    for i, a in enumerate(R1, 1):
        st, note = J[i]
        shot = OUT / f"recheck_{i:02d}_1.png"
        fh.write(json.dumps({"title": a["title"], "status": st, "note": note,
                             "shot": str(shot) if shot.exists() else None}, ensure_ascii=False) + "\n")
print("written", len(R1))
