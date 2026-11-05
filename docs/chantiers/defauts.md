# Ce qu'obtient un nouvel utilisateur : les décisions du 5 novembre

Branche `defauts`. Elle fait suite au verdict Rosette (`docs/chantiers/rosette_verdict.md`), avec les décisions de
Nicolas du 5 novembre au soir. Le contrôle des valeurs par défaut est dans `F:\astro_essais\defauts\`.

## 1. Les échantillons du gradient évitent la nébulosité (fait)

- **L'étape « Relevé H-alpha » revient dans la chaîne** (bloc Astrométrie). Elle télécharge une fois le relevé
  professionnel (NSNS au nord, Finkbeiner ailleurs) et le garde sur le disque. Elle ne bloque jamais un calcul :
  sans réseau, ou sans relevé pour le champ, elle note « aucun » et la raison.
- **Le bloc Gradient lit ce relevé** quand il choisit lui-même ses échantillons. Le réglage « Loin de l'émission »,
  actif par défaut, écarte les zones où le relevé voit de la nébulosité. Le relevé est lissé à l'échelle d'une case,
  pour suivre des zones et non son bruit.
- **Deux gardes** :
  - un relevé qui ne montre que son bruit est ignoré (son étendue doit dépasser 12 fois son bruit) ;
  - s'il ne reste pas assez de ciel, le bloc revient aux échantillons de l'image seule et le dit dans le journal.
- **Ce qui ne change pas** : les échantillons placés à la main sont gardés tels quels, la méthode automatique est
  inchangée, et le modèle du ciel ne lit jamais le relevé.
- **Dans l'éditeur du gradient** (Inspection > Gradient), le relevé s'affiche en ambre par la touche R ou par sa
  bascule dans la barre de l'image. « Générer la grille » respecte « Loin de l'émission ».
- **Résultat sur la Rosette**, avec les seuls réglages par défaut : niveau du rouge au-dessus du ciel, rapporté au
  cœur.

  | Côté | Avant | Maintenant | Relevé |
  |---|---|---|---|
  | Gauche | 0,04 | 0,28 | 0,20 |
  | Haut | 0,34 | 0,56 | 0,35 |
  | Bas | 0,30 | 0,44 | 0,33 |
  | Droite | −0,09 | 0,12 | 0,15 |

## 2. Le curseur « Tons faibles » (fait)

- **Sur chaque calque de workflow du Post**, entre Contraste et Saturation.
- **La méthode** : un étirement hyperbolique généralisé (Payne et Cranfield) de la plage au-dessus du ciel. Son point
  noir et son point de symétrie sont au ciel, comme le placent les utilisateurs de cette méthode (au pied de
  l'histogramme). Il s'applique sur la luminance, en gardant les rapports de couleurs.
- **L'effet** : les tons juste au-dessus du ciel montent le plus, le ciel ne bouge pas, et les hautes lumières
  montent le moins. En dessous de 0, c'est l'inverse.
- **Sur la Rosette**, ce seul curseur remplace la Courbe des tons de 12 points de l'essai 50.
- L'étirement hyperbolique est maintenant partagé entre le moteur (`render/ghs.py`) et le filtre du Post.

## 3 et 4. Les valeurs par défaut, contrôlées sur Heart, Iris, M8 et M31 (fait)

- **M31 rejoint le banc** à partir des images de Nicolas de 2023 (`G:\Astro\2023-08-24_M31_top`, lues sans y écrire),
  avec son rendu `traitement\m31_flat.jpg` comme cible.
- **L'outil `tools/bench/defaults_check.py`** calcule le workflow « Full » aux réglages par défaut sur les quatre
  cibles, puis rend le calque étoiles et fond avec les défauts d'avant et ceux proposés.
- **Le résultat est meilleur sur les quatre** : le champ d'étoiles est moins envahissant, et la nébuleuse ou la
  galaxie ressort ; les extensions et les bras extérieurs montent sans brûler les cœurs.
- **Nouvelles valeurs par défaut**, chacune au milieu de son curseur, l'ancien rendu restant à un cran :
  - force des étoiles 0,6 et taille −1 (les positions 1/1 d'avant) ;
  - Tons faibles 0,8.
- **Constat** : sur M31, notre rendu par défaut reste moins contrasté et moins coloré que celui de Nicolas (bandes de
  poussière pâles). À reprendre, sans toucher à ces réglages.

## 5. Les fichiers de chaque type choisis par l'utilisateur (fait pour l'import)

- **Le dialogue d'import** a une ligne par type : brutes, darks, flats, offsets, darks de flat.
- **Chaque ligne montre** ce qui est choisi (un dossier, ou « n fichiers de… »), le nombre d'images, la pose et l'ISO.
  Elle précise aussi :
  - « trouvés d'après les noms de dossiers » tant que c'est la suggestion ;
  - une pose de darks différente de celle des brutes ;
  - un type qui reprend les images des brutes.
- **Chaque ligne permet** de choisir des fichiers (sélection multiple), de choisir un dossier, de retirer le type
  (✕), ou de déposer des fichiers ou dossiers depuis l'Explorateur.
- **Chaque changement relit le choix** (`api.scan_files`), et la session reçoit exactement ce qui a été choisi.
- **À faire** : les mêmes lignes dans Images > Parcourir et dans le panneau Calibration, pour changer les fichiers
  d'une session déjà importée.

## 6. Le correcteur localisé (fait)

- **Post > Détail > Correcteur.** Un clic sur une tache la corrige.
- **La source** est trouvée aussitôt dans le voisinage (le disque dont le bord ressemble le plus au bord de la tache,
  une étoile y étant pénalisée), puis enregistrée avec la tache : l'aperçu et l'export sont identiques.
- **Le raccord** se fait par édition de Poisson (Pérez et al. 2003, le principe du pinceau correcteur de Georgiev
  2004), avec un bord adouci.
- **Les gestes**, ceux des logiciels photo :
  - glisser un cercle pour déplacer la tache ou sa source ;
  - Alt + clic pour placer la source de la tache sélectionnée ;
  - clic droit pour retirer ;
  - [ ] pour la taille ;
  - chaque geste est un pas d'Annuler.
- **Sur la Rosette**, quatre clics effacent la tache de poussière.

## Restent

- Les lignes de fichiers dans Parcourir et Calibration (fin du point 5).
- Le rapport d'ergonomie de l'agent (`F:\astro_essais\ergonomie`). Plusieurs petits points sont encore ouverts :
  - les calques de workflow ajoutés seuls dans le Post ;
  - un workflow conseillé à la création ;
  - les noms en double ;
  - l'export en JPEG par défaut ;
  - des mots techniques (SCNR, CLAHE, RBF).
- Plus tard, sur décision de Nicolas : une vraie réduction d'étoiles, les traînées diagonales du capteur.
