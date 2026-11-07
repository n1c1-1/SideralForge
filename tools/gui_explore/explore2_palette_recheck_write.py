"""palette_matrix round 2: the verdict lines of the recheck (written after reading recheck_raw.jsonl and the shots)."""
import json
from pathlib import Path

D = Path("F:/astro_essais/A_REGARDER/exploration2/palette_matrix")
R1 = Path("F:/astro_essais/A_REGARDER/exploration/palette_matrix/anomalies.jsonl")
titles = [json.loads(x)["title"] for x in R1.read_text(encoding="utf-8").splitlines() if x.strip()]


def s(n):
    return str(D / f"rc_{n}.png")


V = [
    ("corrigé", "Image posée par double-clic ou par glisser sur la zone du bas : un seul Ctrl+Z revient exactement à "
     "l'état d'avant, et le nouveau workflow « Image 2 » n'a plus de blocs ajoutés d'office ; Ctrl+Y le remet.",
     s("image_undo")),
    ("corrigé", "En tenant la carte Session, Image ou Fusion au-dessus de la zone « + Flux : Session/Image », la zone "
     "s'allume (bord bleu).", s("zone_lit_session")),
    ("corrigé", "Session sans workflow : en glissant Calibration une place « Calibration » s'allume et le dépôt pose le "
     "bloc dans un nouveau workflow.", s("session_sans_wf_glisser")),
    ("toujours là", "À l'ouverture d'un workflow complet un bloc « Recombiner » est déjà dessiné (Post dit déjà « 1 "
     "calque ») ; le premier double-clic sur la carte ne change rien à l'écran, aucun message.", s("combiner_1")),
    ("corrigé", "Deux poses de Recombiner puis Ctrl+Z : le dernier posé disparaît ; Ctrl+Y le remet.", s("combiner_2")),
    ("toujours là", "Les deux blocs s'appellent tous deux « Recombiner », sans numéro (graphe et panneau).",
     s("combiner_2")),
    ("toujours là", "Le nom enregistré du calque reste « Full · Recombine » (anglais) ; Post ne liste les calques "
     "qu'après un calcul, donc l'affichage n'a pas pu être vu à l'écran.", None),
    ("corrigé", "Avec un workflow couleur vide, la carte Fusion est grisée et dit « Requiert 2 flux filtres, ou "
     "empilement couleur » ; le double-clic ne pose rien.", s("fusion_vide")),
    ("corrigé", "Une Session lâchée sur Post, sur le bloc Image ou dans le vide est refusée (aucune session ajoutée).",
     s("session_sur_post")),
    ("corrigé", "Session sur page vierge, et Calibration sur Session seule : un seul Ctrl+Z revient exactement à l'état "
     "d'avant.", s("calibration_ctrlz")),
    ("corrigé", "Projet neuf modèle Complet, Session lâchée sur la zone du bas : pas d'erreur, un second flux "
     "« Workflow » avec la Session 2, l'enregistrement marche, Ctrl+Z revient à l'état d'avant.",
     s("modele_complet_zone")),
    ("corrigé", "Deux flux : Calibration lâchée sur la place du second flux, posée dans « Flow », choisie, palette "
     "« pour « Flow » », aucune erreur.", s("deux_flux_cal_flux2")),
    ("corrigé", "Projet mono, carte Astrométrie tenue : les places « nouvelle branche » sont entre les deux lignes, "
     "plus aucun chevauchement mesuré.", s("mono_branche")),
    ("corrigé", "La zone « + Flux : Session/Image » est sous les blocs de la colonne des sources (Session + Image, deux "
     "sessions, Image posée) : aucun chevauchement mesuré.", s("zone_session_image")),
    ("corrigé", "Workflow complet, Session 2 ajoutée : sa Calibration ne touche plus la zone du bas.", s("zone_cal2")),
    ("corrigé", "Modèle « Image déjà étirée » : source marquée étirée, aucun bloc linéaire, la palette grise Calibration "
     "à Couleur avec une raison (« pas de … depuis image étirée »).", s("modele_etiree")),
    ("changé", "À l'ouverture d'un projet Image seule, le panneau n'est plus vide mais montre « Images / Importer "
     "dossier… » (le panneau des brutes), la palette dit « nouveau workflow » et grise tous les blocs avec « poser "
     "Session/Image d'abord » alors qu'une Image est posée (noté à part).", s("panneau_ouverture")),
    ("corrigé", "Deux flux ou mono : la Session posée par double-clic rejoint le flux choisi (« Workflow » ou « Ha ») "
     "et est dessinée.", s("session_orpheline_two_flows")),
    ("corrigé", "Après la pose de deux Recombiner, Ctrl+Y n'a plus rien à refaire (grisé), le Gradient retiré ne revient "
     "pas.", s("combiner_ctrly")),
    ("corrigé", "Sur une image empilée, Split (« Étoiles/fond ») dit maintenant « Image linéaire : partir des brutes, ou "
     "la marquer étirée » : plus d'impasse vers Débruitage.", s("raison_split")),
    ("corrigé", "Workflow de poses vide : Astrométrie dit « Astrométrie requiert Empilement avant ».",
     s("raison_astrometrie")),
    ("corrigé", "Fusion grisée dit ce qui manque : « Requiert 2 flux filtres, ou empilement couleur ».",
     s("raison_fusion")),
]
assert len(V) == len(titles), (len(V), len(titles))
with (D / "recheck.jsonl").open("w", encoding="utf-8") as fh:
    for t, (st, note, shot) in zip(titles, V):
        fh.write(json.dumps({"title": t, "status": st, "note": note, "shot": shot}, ensure_ascii=False) + "\n")
print({k: sum(1 for v in V if v[0] == k) for k in ("corrigé", "toujours là", "changé")})
