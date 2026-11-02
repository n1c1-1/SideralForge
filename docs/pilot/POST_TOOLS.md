# Outils de Post : inventaire PixInsight / Siril (2026-11-02)

But : finir une image sans Photoshop, PixInsight ni Siril. Règle (ENGINE_KNOWLEDGE 4b) : chaque outil suit une
méthode publiée ou documentée, nommée ; pas de recette maison.

Post travaille sur des calques **déjà étirés** (valeurs d'affichage 0–1). Un outil y a sa place s'il agit sur une
image affichable. Ce qui demande les données **linéaires** reste dans le moteur (ses étapes).

Docs citées : PixInsight `https://pixinsight.com/doc/tools/<Outil>/<Outil>.html` ; Siril 1.4
`https://siril.readthedocs.io/en/latest/processing/<page>.html` (stretching, colors, filters, stars) et le tutoriel GHS
`https://siril.org/tutorials/ghs/`.

## 1. Ce qu'on avait (13 filtres)

Étirement « GHS » (en fait un arcsinh autour d'un point de symétrie), Courbes à 3 curseurs, Niveaux, Saturation par
plage de teintes, SCNR (neutre moyen), Balance des couleurs, Contraste local, Accentuation (masque flou), Réduction du
bruit (flou gaussien), Réduction d'étoiles, Réduction des halos, Couleur des étoiles, H-alpha vers rouge.

## 2. Inventaire

| Outil de référence (doc) | Chez nous | En calque ? |
|---|---|---|
| PI CurvesTransformation ; Siril Curves (stretching) ; Photoshop Courbes | **Courbes** : vrai outil (§3) | oui |
| PI HistogramTransformation ; Siril Histogram transformation (MTF) | **Transformation d'histogramme**, ajoutée | oui |
| PI GHS ; Siril Generalised Hyperbolic Stretch + tutoriel GHS | **Étirement hyperbolique généralisé**, ajouté | oui (le tutoriel l'applique aussi sur l'étiré) |
| PI ArcsinhStretch ; Siril Asinh transformation | **Étirement arcsinh** (version Siril), ajouté | oui |
| PI ExponentialTransformation | manque ; les courbes le font | possible |
| PI LocalHistogramEqualization ; Siril CLAHE (filters) | **CLAHE**, ajouté | oui |
| PI ColorSaturation (courbe par teinte) ; Siril Color saturation (colors) | Saturation par plage + courbes S et c* | oui ; courbe par teinte à faire |
| PI SCNR ; Siril Remove green noise (colors) | **SCNR** : 4 protections + préserver la luminosité | oui |
| PI Invert ; Siril Negative | préréglage Négatif des courbes | oui |
| PI MultiscaleLinearTransform, ATrousWaveletTransform ; Siril Wavelets (filters) | **Accentuation multi-échelle** et **Réduction du bruit multi-échelle**, ajoutées | oui |
| PI MultiscaleMedianTransform, TGVDenoise, ACDNR | manquent | possible (coûteux) |
| PI HDRMultiscaleTransform | manque | possible, mais l'algorithme n'est pas publié en détail |
| PI UnsharpMask | Accentuation (masque flou) | oui |
| PI MorphologicalTransformation + StarMask | Réduction d'étoiles (érosion dans le masque d'étoiles) | oui |
| Siril EPF (bilatéral, guidé), Unpurple (filters) | manquent | possibles |
| Siril Star recomposition | calques étoiles / sans étoiles en mode Écran | oui |
| PI AutoHistogram, MaskedStretch, AdaptiveStretch, STF ; Siril autostretch | premier étirement du linéaire | **non** : moteur (sortie) |
| PI BackgroundNeutralization, ColorCalibration, PCC, SPCC, LinearFit ; Siril étalonnage des couleurs | | **non** : moteur (Couleur) |
| PI DBE/ABE ; Siril Background extraction, GraXpert | | **non** : moteur (Gradient) |
| PI Deconvolution ; Siril Deconvolution ; Siril Noise reduction (Anscombe) ; MureDenoise | | **non** : linéaire, moteur |
| Siril StarNet, Unclip stars, Synthetic stars | séparation étoiles du moteur | **non** : linéaire, moteur |
| PI LRGBCombination, ChannelCombination, HDRComposition ; Siril RGB composition, HDR, PixelMath | | **non** : combinaison de piles, moteur (blocs) |

## 3. Courbes (le plus important)

- Points ajoutés par clic, déplacés au glisser, supprimés par Ctrl+clic, clic droit ou glisser hors du graphe (gestes
  de PixInsight et de Photoshop) ; flèches = 1 niveau 8 bits ; Entrée / Sortie numériques ; histogramme en direct de
  l'entrée du filtre (linéaire ou log).
- Canaux, dans l'ordre de Siril : RVB/K, R, V, B, L* (CIE, couleurs gardées), S (saturation TSV, L* remis : la courbe
  S « HSVL* » de PixInsight), c* (chroma CIE, comme le C de Siril), H (teinte CIE, le H de PixInsight).
- Interpolation : Akima (défaut de PixInsight et de Siril ; Akima, J. ACM 17(4), 1970), spline cubique naturelle,
  linéaire (les trois de Siril 1.4).
- Vitesse : table de 4096 valeurs pendant le glisser ; calcul exact au relâcher.
- Préréglages : Linéaire ; les 9 de Photoshop (points lus dans les fichiers .acv livrés avec Photoshop 2026) ; 4
  classiques astro construits sur des fonctions documentées : « Remonter les tons moyens » (MTF 0,35),
  « Contraste des tons moyens, hautes lumières protégées » (GHS, tutoriel Siril), « Éclaircir, contraste conservé »
  (GHS à b négatif), « Renfort de saturation » (courbe S de PixInsight). Les valeurs de ces 4 sont des choix, la forme
  est celle de la source.
- Préréglages perso : enregistrés à côté des réglages ; import et export .acv (Photoshop) et .json.

## 4. Filtres ajoutés, et pourquoi

| Filtre | Méthode | Vérifié |
|---|---|---|
| Étirement hyperbolique généralisé | équations de Payne et Cranfield (code de Siril), modèles canaux / luminance | = siril-cli 1.4 à 0,000 niveau |
| Étirement arcsinh | Siril (Lupton et al. 2004), couleurs gardées | = siril-cli à 0,000 niveau |
| Transformation d'histogramme | MTF de PixInsight et Siril, par canal | = siril-cli à 0,000 niveau |
| SCNR | 4 protections de Siril et PixInsight + luminosité | = siril-cli à 0,000 niveau |
| CLAHE | Zuiderveld 1994, comme OpenCV (utilisé par Siril), sur L* | = OpenCV à 1 niveau près |
| Accentuation multi-échelle | starlet B3 à trous (Starck et Murtagh), biais par couche (MLT) | reconstruction exacte |
| Réduction du bruit multi-échelle | seuillage doux k·σ par couche (Starck et Murtagh, MLT) | bruit /2 sur image test |

Pourquoi ceux-là : ce sont les outils de finition que PixInsight et Siril ont tous deux et qui manquaient (étirements,
contraste local, ondelettes, SCNR complet). Tous sont neutres par défaut.

Vitesse (projet test Heart, 6216 × 4224) : un brouillon pendant le glisser prend 38 à 77 ms avec le CPU (image au
1/8) et 51 à 191 ms avec le GPU (image au 1/4 ; les filtres eux-mêmes tournent sur le CPU, le GPU sert au rendu des
calques). Courbes et GHS sont les plus rapides, CLAHE et ondelettes les plus lents.

## 5. Laissés de côté

- HDRMultiscaleTransform : algorithme non publié en détail, donc pas de recette maison.
- ColorSaturation par courbe de teinte, EPF, Unpurple, MMT, TGVDenoise : possibles en calque, pas faits faute de
  temps (les courbes S, c* et H couvrent une partie).
- L'ancien étirement « GHS » (arcsinh) n'est plus proposé ; les compositions enregistrées s'ouvrent comme avant.
  L'ancienne « Courbes » à 3 curseurs s'ouvre en courbe linéaire identique.
