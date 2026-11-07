"""post_images: the anomalies reproduced twice (explore5_post_images_*.py, logs in F:/astro_essais/_explore4_work/
post_images/run_*.log), written with their images."""
import numpy as np
from harness import Explorer
from PIL import Image, ImageDraw

ex = Explorer("post_images", lang="fr")
out = ex.out


def side(a, b, la, lb, name, crop=None):
    A, B = np.array(Image.open(out / a)), np.array(Image.open(out / b))
    if crop:
        y0, x0, h, w = crop
        A, B = A[y0:y0 + h, x0:x0 + w], B[y0:y0 + h, x0:x0 + w]
    h = min(A.shape[0], B.shape[0])
    im = Image.fromarray(np.concatenate([A[:h], np.full((h, 8, 3), 255, np.uint8), B[:h]], 1))
    d = ImageDraw.Draw(im)
    d.text((10, 10), la, fill=(255, 255, 0))
    d.text((A.shape[1] + 18, 10), lb, fill=(255, 255, 0))
    im.save(out / name)
    return str(out / name)


OPEN = ["Copie du projet de banc Iris (express) ouverte, onglet Post (vue ajustée, 25 %)"]

ex.note(
    "gênant",
    "Saturation forte : l'aperçu n'a pas les couleurs de l'export ni de la vue à 100 % (fond vert à l'écran, violet "
    "dans le fichier)",
    "Avec un filtre Saturation à +100 % sur le calque « Classic stack » seul, l'image ajustée à l'écran a un fond "
    "vert-brun, le TIFF 16 bits exporté (réduit à la même taille) un fond violet plus clair. Écart moyen par canal "
    "export - écran : R +10, V +1,3, B +13,6 niveaux sur 255 ; fond seul : écran (41, 52, 36), export (55, 54, 55). "
    "Sans filtre l'écart reste faible (R +2,2, V +1,2, B +2,8). Même saut à l'écran en zoomant à 100 % : la zone nette "
    "arrive avec le fond violet (zone nette réduite - image moitié : R +4,6, B +6,0), et pendant un déplacement la "
    "limite entre la zone nette et le reste se voit comme un bord net de couleur (2e image : "
    "recheck_region_edge_just_released.png ; une fois la zone recalculée l'image est uniforme). Reproduit 3 fois (export "
    "de la composition : R +4,9, B +6,3 ; Saturation seule ; zone à 100 %). Probable : le filtre agit sur le bruit "
    "coloré du fond à pleine résolution, pas sur l'image réduite de l'aperçu. Images : x_base_saturation_screen.png "
    "(avant) / x_base_saturation_export_reduced.png (export).",
    shot=side("x_base_saturation_screen.png", "x_base_saturation_export_reduced.png", "ECRAN (vue ajustee)",
              "EXPORT TIFF 16 (reduit a la taille de l'ecran)", "anomalie_saturation_ecran_vs_export.png",
              (100, 1200, 600, 800)),
    steps=OPEN + ["Masquer les autres calques (œil), sélectionner « Classic stack »",
                  "+ Filtre > Couleur > Saturation, tirer « Intensité » à +100 %",
                  "Exporter… (TIFF 16 bits, pleine résolution), ouvrir le fichier et le comparer à l'écran",
                  "Ou : bouton 100 %, puis glisser l'image de 700 px vers la gauche"])

ex.note(
    "question",
    "Les calques des workflows avec Débruitage ont un fond vert, les mêmes sans Débruitage sont neutres",
    "Chaque calque montré seul (œil), réglages suggérés : « Classic stack » et « … automatic gradient » ont un fond "
    "neutre (moyenne R 56,6 / V 56,4 / B 55,6), « … denoise » et « … denoise adapted » un fond nettement vert "
    "(49 / 69 / 55 et 51 / 69 / 58). Les vignettes du panneau le montrent aussi. Ces quatre calques sont marqués "
    "« aperçu » (résultats d'un calcul plus ancien), donc la cause peut être le calcul, pas Post ; mais dans Post le "
    "passage d'un calque à l'autre fait sauter la couleur. Reproduit 2 fois (ouverture, puis calques un par un). "
    "Images : alone_1.png (sans débruitage) / alone_3.png (avec).",
    shot=side("alone_1.png", "alone_3.png", "Classic stack", "Classic stack - denoise", "anomalie_debruitage_vert.png"),
    steps=OPEN + ["Cocher l'œil d'un seul calque à la fois et comparer"])

ex.note(
    "question",
    "Curseurs « 1 px » de Netteté multi-échelle et de Réduction du bruit multi-échelle : rien ne change en vue ajustée",
    "Vue ajustée (25 %), filtre ajouté sur « Classic stack », curseur « 1 px » tiré à 2,6 (netteté) ou 9 σ (bruit) : "
    "l'image à l'écran ne change pas du tout (écart 0,0), sans message. À 100 % la même action change bien l'image "
    "(écart moyen 29 et 12 niveaux). Comme Photoshop (« juger la netteté à 100 % »), mais rien ne le dit à "
    "l'utilisateur, qui croit le curseur sans effet. Reproduit 2 fois (passage de tous les filtres, puis contrôle).",
    shot=str(out / "recheck_ml_sharpen_fit.png"),
    steps=OPEN + ["+ Filtre > Détail > Netteté multi-échelle", "Tirer « 1 px » : l'image ne bouge pas",
                  "Bouton 100 % puis même geste : l'image change"])

ex.note(
    "question",
    "Export d'une composition dont les calques sont des « aperçus » sans le dire",
    "Tous les calques du projet sont marqués « aperçu » dans le panneau (résultats plus anciens que les réglages). "
    "Exporter… écrit le TIFF sans avertissement (ligne d'alerte vide, « 2 fichiers écrits »). L'historique joint "
    "porte la marque, la fenêtre non. Reproduit 3 fois (TIFF 16, JPEG, PNG 16).",
    shot=str(out / "export_dialog_tiff16_done.png"),
    steps=OPEN + ["Exporter… > Exporter"])
print("notes written", len(ex.notes()))
