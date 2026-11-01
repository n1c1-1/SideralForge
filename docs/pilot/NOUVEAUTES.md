# Ce qui a changé dans le logiciel (dimanche 1er novembre)

Tout ce que vous avez demandé ce matin est maintenant **dans le logiciel lui-même**, pas à part. Les captures viennent
du vrai logiciel, sur une copie de test de votre Cœur (Sony A7S, 40 images), prises sans GPU (d'où « CPU seul » en
haut à droite).

## Lancer et quoi ouvrir

- **Lancer** : double-clic sur `client.bat` (dans `C:\Users\Nicolas\sandbox\astro`). Il tourne sur le GPU.
- **Vos 4 projets terminés** : `F:\astro_projets\Horsehead`, `Iris`, `Rosette` et `FlamingStar` (Fichier › Ouvrir
  un projet…). Attention : un projet ouvert enregistre tout seul ce que vous y changez.
- **Pour tout essayer sans risque** : la copie de test `F:\astro_essais\v2\heart_a7s_tour` (celle des captures : son
  Post-traitement garde le calque « Empilement » posé dessus ; décochez-le pour revoir « Full »), et ses sœurs
  `heart_a7s_p11` et `heart_a7s_p12`, intactes.

## Ce que vous avez demandé, et où le voir

**1. Le même visualiseur dans chaque bloc.** Inspection, n'importe quel bloc. Zoom, étirement avec son histogramme,
valeurs des pixels en ADU et en RA/Dec, « Comparer avec » les étapes précédentes, « Exporter… », « Envoyer en
Post-traitement ».

![Visualiseur](nouveautes/01_visualiseur.png)

**2. Les graphiques interactifs.** Inspection › Alignement › onglets « Qualité », « Dérive », « Trajectoire »,
« Répartition » ; Couleur › « Réponse spectrale ». Survol pour lire les valeurs, molette pour zoomer, double-clic pour
revenir, clic sur la légende pour masquer une courbe, « Numéro d'image » ou « Heure » en abscisse, clic droit pour
exporter en CSV ou en PNG. Dans Sélection, un clic sur un point montre l'image dessous.

![Dérive](nouveautes/02_derive.png)

**3. Les images de calibration, une par une, avec leurs mesures.** Inspection › Calibration › onglets « Masters »,
« Darks », « Flats », « Brutes calibrées ». Sous l'image : niveau, bruit, pixels chauds, écart centre-bord ;
flèches ou « Clignoter » pour passer de l'une à l'autre ; « Exclure » retire un dark ou un flat de son master.

![Darks](nouveautes/03_darks.png)

**4. L'ajustement des couleurs étoile par étoile, contre Gaia.** Inspection › Couleur › « Couleurs des étoiles ».
Chaque étoile : sa couleur mesurée contre celle du catalogue, utilisée ou rejetée, et pourquoi (tableau dessous).

![Couleurs des étoiles](nouveautes/04_couleurs_etoiles.png)

**5. La carte de rejet et le poids de chaque image.** Inspection › Empilement › « Carte de rejet », « Couverture »,
« Poids » (graphique et tableau : poids, FWHM, transparence de chaque image).

![Poids](nouveautes/05_poids.png)

**6. La Sortie montre l'image finale.** Inspection › Sortie : le rendu tel que le Post-traitement le montre, et non
plus un aperçu.

![Sortie](nouveautes/06_sortie.png)

**7. L'astrométrie sur votre empilement, avec la grille RA/Dec.** Inspection › Astrométrie › « Empilement et
solution » ; boutons « Grille du ciel » et « Étoiles du catalogue ». Le relevé H-alpha est l'onglet suivant.

![Astrométrie](nouveautes/07_astrometrie.png)

**8. Post-traitement : des calques comme dans Affinity.** Un filtre ajouté se range **sous son calque** (« S'applique
à "Full" seulement ») ou en ligne à part (« Tout ce qui est en dessous ») ; glisser-déposer comme dans Affinity. Les
filtres partent **neutres** (tout à 0) : ils ne changent rien tant que vous ne les réglez pas.

![Calques](nouveautes/08_calques.png)

**9. Export FITS, TIFF, PNG ou JPEG.** Post-traitement › « Exporter… » (et le bouton « Exporter… » de chaque
visualiseur d'Inspection). Le FITS 32 bits garde la solution astrométrique dans son en-tête (relu par Siril).

![Export](nouveautes/09_export.png)

**10. Envoyer une image d'Inspection en Post-traitement.** Bouton « Envoyer en Post-traitement » au-dessus de
n'importe quelle image : elle devient un calque, en haut de la pile (mode de fusion, opacité, masque, filtres ; à 100 % elle cache ce qui est
dessous). Une image encore aux
couleurs de l'appareil arrive avec ses canaux étirés séparément (« Chaque canal »), donc un ciel neutre.

![Calque image](nouveautes/10_calque_image.png)

**11. Les points d'arrêt.** Graphe : clic droit sur un bloc › « Pause ici pour vérification » (pastille jaune sur le
bloc). Le calcul s'arrête après ce bloc et la fenêtre d'exécution attend : « Ouvrir dans Inspection » pour
vérifier, puis « Continuer ».

![Pause](nouveautes/11_pause_graphe.png)

![Continuer](nouveautes/11_pause_fenetre.png)

**12. La durée sur GPU et sur CPU.** Graphe : cliquez Débruitage, Étoiles ou Gradient ; chaque carte de méthode, à
droite, donne les deux durées, celle de cette machine en gras.

![Méthodes](nouveautes/12_methodes.png)

**13. Plus de fenêtres de console** qui s'ouvrent pendant un calcul (rien à voir : elles ne viennent plus).

## Corrigé pendant la vérification finale

- Post-traitement : la **dernière** modification (un filtre, un calque, un renommage) n'était enregistrée qu'à la
  suivante ; elle se perdait à la fermeture du projet ou au changement de langue.
- Une image envoyée depuis Inspection arrivait **verte** en Post-traitement (couleurs de l'appareil).
- Sous les images de calibration, les chiffres de l'image précédente restaient affichés et se chevauchaient.
- La liste des calques s'ouvrait légèrement défilée, la première ligne coupée en haut.
- Le nom d'une image envoyée en Post-traitement suit maintenant la langue.

## Limites connues

- **Résultats calculés avant aujourd'hui** (vos 4 projets compris) : la carte de rejet, la normalisation des poids
  et la rondeur des étoiles n'y sont pas. L'écran le dit : « … pas dans ce résultat, fait par une version plus
  ancienne du moteur. Recalculez Empilement (ou Alignement) pour en disposer. »
- **Post-traitement sur le CPU** : l'aperçu pendant que vous bougez un curseur est rapide (0,15 à 0,45 s), mais
  l'image nette à mi-résolution, quand vous lâchez le curseur, prend encore 3 à 9 s (surtout les curseurs des
  étoiles et du détail).
- Inspection › Sortie : environ 15 s sur le CPU la première fois (le rendu de l'image finale).
- Défauts du moteur que ces écrans **montrent** (corrections prévues après le banc d'essai) : la couleur rejette
  comme « saturées » beaucoup d'étoiles brillantes (G ≈ 7,5) ; « Pixels rejetés 1,8 % » compte aussi les pixels hors
  du champ (sur le Cœur, le vrai rejet est d'environ 0,04 %) ; le nombre de pixels chauds des darks baisse au fil de
  la série (le capteur refroidit : darks peut-être mal appariés).
- Après « Continuer », quand il ne reste rien à calculer, la fenêtre dit seulement « Rien à exécuter ».
