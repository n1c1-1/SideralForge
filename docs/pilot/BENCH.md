# Banc de qualité : notre moteur face à Siril 1.4

*Premier passage, 2026-11-01. Les mesures et les planches se refont avec `tools/bench/` (voir la fin).*

## En bref

<!-- enbref -->
Sur les mêmes fichiers, notre « Classic stack » fait **jeu égal avec Siril 1.4 de base sur la profondeur et la
finesse** (M8, le jeu du tutoriel Siril : égalité presque parfaite). Il le **bat nettement sur la fidélité des couleurs
des étoiles** (1,3 à 3 fois moins d'erreur face à Gaia, sauf l'indice bleu−rouge de M51) et, sur trois cas,
sur le rendu.

Il n'est **pas encore « pas moins bien partout »** :
- **Iris : 15 à 20 % de bruit en plus**, donc 0,15 à 0,2 magnitude de profondeur et environ 20 % de signal/bruit sur
  la nébuleuse en moins. Une partie vient de nos poids de poses.
- **Bruit fin** : à petite échelle, notre bruit est souvent 0,1 à 0,2 mag plus fort, surtout en rouge et en bleu.
  Ailleurs qu'à Iris, il est égal à plus grande échelle.
- **M31 au Seestar** : notre modèle de fond efface une bonne partie du grand disque, et le ciel est violacé.
- **Défauts à 100 %** :
  - les étoiles saturées ont un bord bleu-cyan (M8, Iris) ;
  - il reste de petits points colorés de pixels chauds (M8) ;
  - les étoiles très petites sont un peu moins rondes au centre avec le drizzle (Heart, Seestar).
- **Lenteur** : sur Express, notre calcul est 1,4 à 20 fois plus long que Siril (M8 : 81 min contre 4). Presque tout
  ce temps est du réseau (astrométrie et catalogues avec VizieR muet, plus un catalogue d'étoiles faibles inutile en
  « Classic »). Sur Base, avec les catalogues déjà là, l'écart va de 0,9 à 3,6 fois.
<!-- /enbref -->

## Le banc

Cinq cas, sans recadrage, avec exactement les mêmes fichiers pour les deux logiciels (des copies figées sous
`F:/astro_essais/bench/<cas>/data/`, la liste dans `manifest.json`) :

| Cas | Appareil et instrument | Express | Base | Calibration |
|---|---|---|---|---|
| heart | Sony A7S défiltré, lunette TS61 (272 mm) | 30 poses de 721 | 100 | 28 darks, 46 flats, pas d'offsets |
| iris | Sony A7S défiltré, Newton 130/900 | 30 de 1268 | 100 | 48 darks, 62 flats, pas d'offsets |
| m51 | Canon 6D, Newton 200/1000 | 30 de 159 | 100 | 25 darks, 12 flats, 10 offsets |
| m8 | ZWO ASI2600MC (le jeu du tutoriel Siril) | 15 de 15 | = Express | 15 darks, 15 flats, 15 offsets |
| seestar | Seestar S50, M31 | 30 de 360 | 100 | aucune (le Seestar n'en fait pas) |

Les poses sont prises à intervalles réguliers sur toute la séance. Pour M8, seules les 15 poses à large bande sont
prises : les 20 poses bi-bande (HOO) ont un autre filtre et d'autres flats, et une pile classique ne les mélange pas.
Le niveau Base de M8 est donc le même que l'Express.

## La recette Siril (à valider)

Siril 1.4.0-beta3 en ligne de commande (`siril-cli`), le script officiel « OSC_Preprocessing » de Siril 1.4, puis le
traitement de base qu'un utilisateur de Siril enchaîne. Pas de GraXpert, pas d'IA, pas de retrait d'étoiles. Les
scripts écrits pour chaque cas sont gardés à côté des résultats (`siril_<niveau>/pre.ssf` et `post.ssf`).

1. **Conversion** des fichiers bruts (RAW ou FITS) en séquences Siril.
2. **Maître d'offset** : moyenne des offsets avec rejet winsorisé 3/3, sans normalisation. Sans offsets (les deux cas
   Sony A7S) : un offset synthétique au niveau de noir du capteur (512), comme Siril le conseille pour les APN.
3. **Maître de flat** : chaque flat moins l'offset, puis moyenne winsorisée 3/3 avec normalisation multiplicative.
4. **Maître de dark** : moyenne winsorisée 3/3 sans normalisation.
5. **Calibration des poses** : moins le dark, divisées par le flat (couleurs du flat égalisées), pixels chauds
   corrigés d'après le dark (3 sigma), puis dématriçage (RCD, le réglage par défaut).
6. **Alignement** global sur les étoiles (homographie, interpolation Lanczos, les réglages par défaut).
7. **Empilement** : moyenne avec rejet winsorisé 3/3, normalisation additive avec mise à l'échelle, fonds RVB égalisés,
   résultat en 32 bits. Toutes les poses sont gardées : le script officiel n'en exclut aucune. Pour le Seestar, le
   script Seestar officiel de Siril : alignement en deux passes, et les poses aux étoiles les moins rondes écartées.
8. **Astrométrie** par Siril, avec le centre approximatif et la focale (les mêmes que ceux donnés à notre moteur).
9. **Extraction du fond** : polynôme de degré 2 (le même degré que notre bloc Dégradé), 20 échantillons par ligne,
   tolérance 1.
10. **Couleur** :
    - **SPCC** (étalonnage spectrophotométrique, sur les spectres Gaia) quand le capteur est dans la base de Siril :
      ASI2600MC (capteur Sony IMX571) et Seestar S50, avec un filtre UV/IR (le Seestar le dit dans ses fichiers ; pour
      l'ASI2600MC c'est l'usage normal en large bande). Référence de blanc : celle par défaut, « galaxie spirale
      moyenne ».
    - **PCC** (étalonnage photométrique sur les couleurs Gaia) sinon : le Sony A7S défiltré et le Canon 6D ne sont pas
      dans la base, et la courbe d'un capteur défiltré n'est de toute façon pas celle d'origine.
11. **Étirement automatique lié** (les trois couleurs ensemble, comme Siril le demande après un étalonnage des
    couleurs), enregistré en TIFF 16 bits et en PNG.

Deux écarts au « tout par défaut », imposés par le jour du test :
- **Catalogues** : Siril va chercher ses étoiles sur VizieR, qui ne répondait pas le 1er novembre. Siril recommande
  lui-même ses extraits locaux de Gaia DR3 ; ils sont installés pour le banc (1,5 Go pour l'astrométrie et la PCC,
  les morceaux du catalogue spectral qui couvrent les cinq champs pour la SPCC), dans
  `F:/astro_essais/bench/_siril/catalogues`, pas dans le profil de l'utilisateur.
- **Réglages de Siril** : Siril lit une copie des réglages de l'utilisateur (`_siril/config.ini`), pour ne jamais
  écrire les siens. Il calcule avec 8 cœurs sur 16, pour laisser de la place aux autres travaux.

## Nos réglages

Le flux « Classic stack » de notre moteur, réglages suggérés, sans rien toucher : calibration, alignement, empilement,
astrométrie, couleur, dégradé polynomial (degré 2), sortie simple. Le projet reçoit le même centre approximatif et la
même focale que Siril. Il tourne sur la carte cuda:0 ; son état (profils d'appareil copiés, cache des catalogues) est
dans `F:/astro_essais/bench/_home`, jamais dans le dossier de l'utilisateur.

Ce que nos réglages suggérés ont choisi :
- **drizzle** : échelle 1 (pas d'agrandissement) à 30 poses ; 1,5 à 100 poses pour Heart, Iris et le Seestar,
  dont les étoiles sont petites ; 1 pour M51 (étoiles larges) ;
- **sélection** : une pose écartée sur Iris (29/30 et 99/100) et sur Heart Base (99/100) ; Siril garde tout, sauf le
  script Seestar (29/30 et 93/100) ;
- **poids des poses** : selon la transparence, le bruit et la finesse. Sur Iris, ils vont de 0,2 à 2,1 fois la
  médiane : l'équivalent de 86 poses égales sur 99. Sur les autres cas, ils restent entre 0,3 et 1,4, soit 96 à 97 % des
  poses.

## Les mesures

Chaque image garde sa propre grille (la nôtre est parfois agrandie par le drizzle, celle de Siril non) : tout est placé
sur le ciel grâce à l'astrométrie de chaque logiciel, et les tailles sont en secondes d'arc. Seul le ciel couvert par
les deux images compte. Les étoiles de référence sont celles de Gaia DR3.

Pour comparer des logiciels qui ne travaillent pas dans les mêmes unités, chaque couleur des deux piles est d'abord
ramenée à la même échelle photométrique (une étoile de magnitude G donne le même flux partout). La luminance est la
moyenne des trois couleurs ainsi ramenées.

Sur les **piles linéaires** (notre pile brute, la pile de Siril avant extraction du fond) :
- **bruit du fond** : l'écart des moyennes de petites boîtes de ciel (étoiles masquées), traduit en magnitude par
  seconde d'arc² ; plus le chiffre est haut, moins il y a de bruit. Mesuré à deux échelles : boîtes de 4 pixels de
  Siril, et 4 fois plus grandes (là, le lissage que l'interpolation de Siril donne au bruit pèse moins) ;
- **profondeur** : la magnitude de l'étoile qu'on détecte à 5 sigma, avec le bruit mesuré dans des ouvertures vides ;
- **étoiles faibles** : combien d'étoiles Gaia isolées ressortent à plus de 5 sigma, et la magnitude où on en trouve la
  moitié ;
- **signal/bruit sur la cible** : la région étendue la plus brillante (nébuleuse ou galaxie) contre le ciel juste
  autour, divisé par le bruit dans la région ;
- **étoiles** : finesse (largeur à mi-hauteur, en secondes d'arc) et rondeur (petit axe / grand axe), au centre et au
  bord du champ, sur des étoiles isolées non saturées ;
- **traînées et défauts du rejet** : traînées longues et fines restantes, et petites taches claires ou sombres qui ne
  sont ni des étoiles Gaia ni présentes dans l'autre image (pixels chauds, rayons cosmiques mal rejetés). Le
  détecteur de traînées a été essayé sur une traînée artificielle : il la trouve à partir de 4 sigma par pixel, pas
  en dessous.

Sur les **images linéaires finales** (fond retiré, couleurs étalonnées) :
- **planéité du fond** : le « plancher » de 8×8 zones (étoiles masquées) et son écart, en magnitude par seconde d'arc²
  (plus haut = plus plat). Attention : dans un champ rempli de nébulosité, une partie de cet écart est la nébulosité
  elle-même ;
- **couleurs des étoiles** : l'indice bleu−rouge (et vert−rouge) de chaque étoile face à sa couleur Gaia BP−RP.
  L'erreur est la dispersion autour d'une droite, ramenée en unités de BP−RP (plus bas = plus fidèle). La pente dit la
  saturation des couleurs, le point blanc dit quelle étoile ressort blanche (le Soleil : BP−RP = 0,82).

Sur les **images finales** : une planche par cas, Siril à gauche, nous à droite, la même zone du ciel (notre image
remise dans le cadrage de Siril par l'astrométrie), et quatre recadrages à 100 % : la cible, le centre, un coin et une
étoile brillante. Les deux rendus par défaut ne visent pas le même fond de ciel : Siril éclaire le fond (gris clair),
nous le gardons sombre ; la planche sert à juger les défauts, pas le goût.

## Résultats par cas

Chaque tableau : notre valeur, celle de Siril, et qui fait mieux (« égal » quand l'écart est sous le seuil de la
mesure : 0,05 mag, 3 à 5 %). Sous les tableaux, la planche des images finales (Siril à gauche).

### Heart (Sony A7S défiltré, lunette 272 mm)

<!-- verdict:heart -->
**Égalité sur la profondeur, avantage à nous sur la finesse et les couleurs.**

- **Express (30 poses)** : Siril a un peu moins de bruit (0,13 mag à petite échelle, 0,05 à grande échelle) et va
  0,14 mag plus profond ; le nombre d'étoiles faibles trouvées est le même (−1 %). Le signal/bruit sur le cœur de la
  nébuleuse est meilleur de 12 % chez Siril.
- **Base (100 poses, notre pile agrandie 1,5 fois)** : profondeur, étoiles faibles et signal/bruit sont égaux. Le
  bruit est un peu plus fort chez nous à petite échelle, un peu plus faible à grande échelle.
- **Étoiles plus fines chez nous** : 11,7″ contre 13,3″ au centre en Base, et plus rondes au bord. Mais un peu moins
  rondes au centre une fois le drizzle à 1,5 (0,87 contre 0,92), comme sur le Seestar.
- **Couleurs des étoiles** trois fois plus fidèles (erreur 0,06-0,08 contre 0,17-0,33 en BP−RP).
- **Planche** : notre H-alpha est rouge franc, celui de Siril rose pâle sur un fond gris. Le fond est un peu plus plat
  chez Siril en Base.
- **Défaut à noter** : dans les coins, nos étoiles faibles ont souvent un bord bleu d'un côté et rouge de l'autre ;
  elles sont plus neutres chez Siril. Nos couleurs d'étoiles plus saturées (pente 1,3 contre 0,4) rendent ce défaut
  plus visible.
<!-- /verdict -->

**Express (30 poses)**

<!-- table:express:heart -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 30 | 30 | |
| Bruit du fond (mag/arcsec² sur 25.5″, plus haut = moins de bruit) | 24.48 | 24.61 | Siril |
| Bruit du fond à plus grande échelle (102″) | 25.24 | 25.29 | Siril |
| Profondeur : étoile détectable à 5σ (magnitude G) | 16.23 | 16.37 | Siril |
| Étoiles faibles trouvées à 5σ (sur 26489 étoiles Gaia isolées) | 23191 | 23440 | égal |
| Magnitude où la moitié des étoiles est trouvée | 15.69 | 15.73 | égal |
| Rapport signal/bruit sur la cible | 4.4 | 5.0 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 12.14 | 13.13 | **nous** |
| FWHM au bord du champ (″) | 14.86 | 15.06 | égal |
| Rondeur au centre (1 = rond) | 0.90 | 0.91 | égal |
| Rondeur au bord | 0.84 | 0.78 | **nous** |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 0 | 0 | égal |
| Taches sombres propres à l'image (par degré²) | 0 | 0 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 24.30 | 24.38 | égal |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.084 | 0.218 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.169 | 0.326 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.38 | 0.41 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.89 | 0.94 | |
| Cible : signal / bruit en mag/arcsec² (info) | 23.49 / 25.09 | 23.55 / 25.28 | |
| Durée (min) | 21.8 | 2.3 | |
<!-- /table -->

![heart express](bench/heart_express.jpg)

**Base (100 poses)**

<!-- table:base:heart -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 99 | 100 | |
| Bruit du fond (mag/arcsec² sur 25.5″, plus haut = moins de bruit) | 24.73 | 24.82 | Siril |
| Bruit du fond à plus grande échelle (102″) | 25.35 | 25.28 | **nous** |
| Profondeur : étoile détectable à 5σ (magnitude G) | 16.64 | 16.63 | égal |
| Étoiles faibles trouvées à 5σ (sur 26553 étoiles Gaia isolées) | 24456 | 24350 | égal |
| Magnitude où la moitié des étoiles est trouvée | 16.00 | 15.96 | égal |
| Rapport signal/bruit sur la cible | 6.2 | 6.4 | égal |
| Finesse des étoiles, FWHM au centre (″) | 11.74 | 13.28 | **nous** |
| FWHM au bord du champ (″) | 14.41 | 15.26 | **nous** |
| Rondeur au centre (1 = rond) | 0.87 | 0.92 | Siril |
| Rondeur au bord | 0.83 | 0.77 | **nous** |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 0 | 0 | égal |
| Taches sombres propres à l'image (par degré²) | 0 | 0 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 24.56 | 24.70 | Siril |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.056 | 0.172 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.083 | 0.255 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.34 | 0.43 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.91 | 0.91 | |
| Cible : signal / bruit en mag/arcsec² (info) | 23.41 / 25.40 | 23.53 / 25.55 | |
| Durée (min) | 9.4 | 5.6 | |
<!-- /table -->

![heart base](bench/heart_base.jpg)

### Iris (Sony A7S défiltré, Newton 130/900)

<!-- verdict:iris -->
**Siril gagne sur le bruit, nous sur les couleurs.** À 30 comme à 100 poses, notre pile a 15 à 20 % de bruit en plus
(−0,15 à −0,20 mag/arcsec²), aux petites comme aux grandes échelles. Siril va donc un peu plus profond : +0,15 à
+0,19 mag, 3 à 4 % d'étoiles faibles en plus. Le rapport signal/bruit sur la nébuleuse est meilleur de 17 à 24 %
chez Siril. Le signal est le même, c'est le bruit qui diffère.
- Une partie de l'écart s'explique : nos poids de poses vont de 0,2 à 2,1 fois la médiane, ce qui revient à
  86 poses utiles sur 99 (7 à 9 % de bruit en plus), sans étoiles plus fines en échange.
- Le rouge et le bleu sont aussi plus bruités à petite échelle (voir le verdict).
- Le reste n'est pas expliqué.

Finesse et rondeur des étoiles sont égales. Les couleurs des étoiles sont plus fidèles chez nous (erreur 0,035 contre
0,051 en BP−RP). Le fond est égal en Base, plus plat chez nous en Express. Sur la planche, notre nébuleuse est bleue
et structurée, celle de Siril blanc-gris. En Express, le cœur de l'étoile centrale (HD 200775) a chez nous un point
cyan, disparu en Base.
<!-- /verdict -->

**Express (30 poses)**

<!-- table:express:iris -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 29 | 30 | |
| Bruit du fond (mag/arcsec² sur 7.7″, plus haut = moins de bruit) | 23.67 | 23.87 | Siril |
| Bruit du fond à plus grande échelle (31″) | 25.10 | 25.24 | Siril |
| Profondeur : étoile détectable à 5σ (magnitude G) | 17.26 | 17.45 | Siril |
| Étoiles faibles trouvées à 5σ (sur 14462 étoiles Gaia isolées) | 4780 | 4971 | Siril |
| Magnitude où la moitié des étoiles est trouvée | 16.79 | 16.88 | Siril |
| Rapport signal/bruit sur la cible | 2.2 | 2.9 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 6.18 | 6.15 | égal |
| FWHM au bord du champ (″) | 7.91 | 7.95 | égal |
| Rondeur au centre (1 = rond) | 0.88 | 0.89 | égal |
| Rondeur au bord | 0.64 | 0.64 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 8 | 5 | égal |
| Taches sombres propres à l'image (par degré²) | 4 | 3 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 23.23 | 22.95 | **nous** |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.044 | 0.057 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.060 | 0.071 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.52 | 0.67 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.91 | 0.87 | |
| Cible : signal / bruit en mag/arcsec² (info) | 23.52 / 24.39 | 23.44 / 24.59 | |
| Durée (min) | 7.9 | 3.2 | |
<!-- /table -->

![iris express](bench/iris_express.jpg)

**Base (100 poses)**

<!-- table:base:iris -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 99 | 100 | |
| Bruit du fond (mag/arcsec² sur 7.7″, plus haut = moins de bruit) | 24.38 | 24.54 | Siril |
| Bruit du fond à plus grande échelle (31″) | 25.66 | 25.84 | Siril |
| Profondeur : étoile détectable à 5σ (magnitude G) | 17.95 | 18.10 | Siril |
| Étoiles faibles trouvées à 5σ (sur 15526 étoiles Gaia isolées) | 6597 | 6818 | Siril |
| Magnitude où la moitié des étoiles est trouvée | 17.46 | 17.53 | Siril |
| Rapport signal/bruit sur la cible | 3.5 | 4.2 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 5.96 | 6.05 | égal |
| FWHM au bord du champ (″) | 7.39 | 7.53 | égal |
| Rondeur au centre (1 = rond) | 0.88 | 0.90 | égal |
| Rondeur au bord | 0.65 | 0.66 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 7 | 7 | égal |
| Taches sombres propres à l'image (par degré²) | 4 | 5 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 23.34 | 23.40 | égal |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.035 | 0.051 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.039 | 0.051 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.50 | 0.67 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.91 | 0.87 | |
| Cible : signal / bruit en mag/arcsec² (info) | 23.73 / 25.10 | 23.75 / 25.31 | |
| Durée (min) | 8.9 | 6.5 | |
<!-- /table -->

![iris base](bench/iris_base.jpg)

### M51 (Canon 6D, Newton 200/1000)

<!-- verdict:m51 -->
**Presque à égalité, avec un rendu nettement meilleur chez nous.** La profondeur, les étoiles faibles, la finesse
et la rondeur sont égales ou à 3 % près. À petite échelle (5″), notre bruit est plus fort de 0,12 mag (environ
12 %). D'où un signal/bruit plus bas sur la galaxie : −15 à −18 %, à signal égal. À 22″, le bruit est égal.

Le fond est plus plat chez nous. Siril laisse une zone bleutée sur la moitié droite : son polynôme de degré 2 ne l'a
pas enlevée. Siril laisse aussi des taches claires (pixels chauds mal rejetés) : 14 à 33 par degré² chez Siril,
1 à 2 chez nous. Les couleurs des étoiles sont égales en bleu−rouge, meilleures chez nous en vert−rouge. Avec la PCC,
Siril rend blanche une étoile de BP−RP 0,35-0,42 : les étoiles de type solaire sortent jaune-orangé. Sur la planche,
nos bras sont bleus, avec un cœur jaune et des régions HII visibles ; la galaxie de Siril est grise.
<!-- /verdict -->

**Express (30 poses)**

<!-- table:express:m51 -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 30 | 30 | |
| Bruit du fond (mag/arcsec² sur 5.4″, plus haut = moins de bruit) | 24.54 | 24.66 | Siril |
| Bruit du fond à plus grande échelle (22″) | 25.89 | 25.91 | égal |
| Profondeur : étoile détectable à 5σ (magnitude G) | 18.22 | 18.26 | égal |
| Étoiles faibles trouvées à 5σ (sur 3603 étoiles Gaia isolées) | 1261 | 1280 | égal |
| Magnitude où la moitié des étoiles est trouvée | 17.59 | 17.68 | Siril |
| Rapport signal/bruit sur la cible | 3.9 | 4.6 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 6.94 | 6.87 | égal |
| FWHM au bord du champ (″) | 8.40 | 8.31 | égal |
| Rondeur au centre (1 = rond) | 0.84 | 0.82 | égal |
| Rondeur au bord | 0.65 | 0.65 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 1 | 14 | **nous** |
| Taches sombres propres à l'image (par degré²) | 3 | 1 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 23.09 | 22.94 | **nous** |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.026 | 0.022 | égal |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.032 | 0.048 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.45 | 0.80 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.88 | 0.42 | |
| Cible : signal / bruit en mag/arcsec² (info) | 23.54 / 25.01 | 23.50 / 25.17 | |
| Durée (min) | 4.0 | 2.9 | |
<!-- /table -->

![m51 express](bench/m51_express.jpg)

**Base (100 poses)**

<!-- table:base:m51 -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 100 | 100 | |
| Bruit du fond (mag/arcsec² sur 5.4″, plus haut = moins de bruit) | 25.18 | 25.30 | Siril |
| Bruit du fond à plus grande échelle (22″) | 26.44 | 26.48 | égal |
| Profondeur : étoile détectable à 5σ (magnitude G) | 18.73 | 18.80 | Siril |
| Étoiles faibles trouvées à 5σ (sur 3778 étoiles Gaia isolées) | 1575 | 1625 | Siril |
| Magnitude où la moitié des étoiles est trouvée | 18.01 | 18.07 | Siril |
| Rapport signal/bruit sur la cible | 2.8 | 3.4 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 7.05 | 7.03 | égal |
| FWHM au bord du champ (″) | 8.46 | 8.47 | égal |
| Rondeur au centre (1 = rond) | 0.83 | 0.81 | égal |
| Rondeur au bord | 0.66 | 0.66 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 2 | 33 | **nous** |
| Taches sombres propres à l'image (par degré²) | 2 | 2 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 23.19 | 23.05 | **nous** |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.026 | 0.018 | Siril |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.028 | 0.041 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.46 | 0.80 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.88 | 0.35 | |
| Cible : signal / bruit en mag/arcsec² (info) | 24.50 / 25.63 | 24.43 / 25.75 | |
| Durée (min) | 8.4 | 9.6 | |
<!-- /table -->

![m51 base](bench/m51_base.jpg)

### M8 et M20 (ASI2600MC, le jeu du tutoriel Siril)

<!-- verdict:m8 -->
**Égalité sur presque tout ; deux défauts visibles chez nous.**

À 15 poses (le jeu complet du tutoriel Siril), les deux images sont égales sur :
- le bruit, aux deux échelles ;
- la profondeur et les étoiles faibles (25 768 contre 25 806) ;
- la finesse et la rondeur des étoiles ;
- la planéité du fond.

Le signal/bruit sur le cœur de la Lagune est 6 % plus bas chez nous. Les couleurs des étoiles sont plus fidèles chez
nous (erreur 0,05 contre 0,07-0,13), bien que Siril ait ici sa SPCC avec le bon capteur. Les deux rendus sont proches ;
le nôtre a des étoiles plus colorées et une H-alpha un peu plus rouge.

Deux défauts chez nous, absents chez Siril :
- **les étoiles brillantes saturées ont un croissant bleu-cyan sombre** sur un bord. On voit le même point cyan au cœur
  de l'étoile centrale d'Iris en Express ;
- **de petits points d'une seule couleur** (bleus ou rouges, 1 à 2 pixels) dans le fond : 57 par degré² contre 7.
  Ce sont des pixels chauds d'un seul canal que la calibration et le rejet laissent passer. Siril les retire avec sa
  correction d'après le dark.

Un reste de traînée faible est aussi visible chez nous à un endroit ; le détecteur ne l'a pas compté, car il est
morcelé par les étoiles.

Le temps : 81 min chez nous, contre 4 min pour Siril.
- 66 min de catalogues, dont environ 45 min pour les étoiles faibles, que « Classic » n'utilise pas ;
- 10 min d'astrométrie, perdues pour l'essentiel à attendre VizieR.
Sans le réseau, notre calcul prend environ 5 min.
<!-- /verdict -->

**Express = Base (15 poses)**

<!-- table:express:m8 -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 15 | 15 | |
| Bruit du fond (mag/arcsec² sur 8.0″, plus haut = moins de bruit) | 23.92 | 23.96 | égal |
| Bruit du fond à plus grande échelle (32″) | 24.07 | 24.09 | égal |
| Profondeur : étoile détectable à 5σ (magnitude G) | 17.54 | 17.59 | égal |
| Étoiles faibles trouvées à 5σ (sur 34086 étoiles Gaia isolées) | 25768 | 25806 | égal |
| Magnitude où la moitié des étoiles est trouvée | 16.67 | 16.68 | égal |
| Rapport signal/bruit sur la cible | 42.8 | 45.4 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 5.50 | 5.62 | égal |
| FWHM au bord du champ (″) | 5.91 | 5.92 | égal |
| Rondeur au centre (1 = rond) | 0.92 | 0.93 | égal |
| Rondeur au bord | 0.89 | 0.87 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 57 | 7 | Siril |
| Taches sombres propres à l'image (par degré²) | 1 | 1 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 23.08 | 23.16 | égal |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.047 | 0.072 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.051 | 0.128 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.38 | 0.70 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.91 | 0.97 | |
| Cible : signal / bruit en mag/arcsec² (info) | 20.49 / 24.57 | 20.48 / 24.63 | |
| Durée (min) | 81.4 | 4.2 | |
<!-- /table -->

![m8 express](bench/m8_express.jpg)

### M31 au Seestar S50

<!-- verdict:seestar -->
**À 100 poses, à égalité ou mieux chez nous ; mais visuellement Siril gagne.** En Express, Siril est un peu
moins bruité (0,1 mag) et trouve 3 % d'étoiles faibles en plus. En Base (notre pile agrandie 1,5 fois par le
drizzle), le bruit est égal à petite échelle, meilleur chez nous à grande échelle, et la profondeur meilleure de
0,11 mag. Les étoiles sont un peu moins rondes chez nous au centre (0,86 contre 0,89-0,91), un peu plus au bord. Les
couleurs des étoiles sont deux à trois fois plus fidèles chez nous.

Défaut visible : notre ciel est violacé et le grand disque de M31, qui remplit le champ, est très atténué ; Siril
montre le disque et les bandes de poussière jusqu'au bord. C'est le problème connu « objet qui remplit le champ » : le
modèle de fond prend le disque pour du ciel. Le fond de Siril est aussi mesuré plus plat.
<!-- /verdict -->

**Express (30 poses)**

<!-- table:express:seestar -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 30 | 29 | |
| Bruit du fond (mag/arcsec² sur 9.5″, plus haut = moins de bruit) | 22.52 | 22.63 | Siril |
| Bruit du fond à plus grande échelle (38″) | 23.67 | 23.78 | Siril |
| Profondeur : étoile détectable à 5σ (magnitude G) | 15.94 | 16.03 | Siril |
| Étoiles faibles trouvées à 5σ (sur 3858 étoiles Gaia isolées) | 650 | 672 | Siril |
| Magnitude où la moitié des étoiles est trouvée | 15.72 | 15.73 | égal |
| Rapport signal/bruit sur la cible | 21.2 | 21.6 | égal |
| Finesse des étoiles, FWHM au centre (″) | 5.93 | 5.83 | égal |
| FWHM au bord du champ (″) | 5.87 | 5.88 | égal |
| Rondeur au centre (1 = rond) | 0.86 | 0.89 | Siril |
| Rondeur au bord | 0.91 | 0.85 | **nous** |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 0 | 2 | égal |
| Taches sombres propres à l'image (par degré²) | 2 | 0 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 20.99 | 21.29 | Siril |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.071 | 0.156 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.137 | 0.257 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.59 | 0.47 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.90 | 0.98 | |
| Cible : signal / bruit en mag/arcsec² (info) | 19.86 / 23.18 | 19.93 / 23.26 | |
| Durée (min) | 4.0 | 0.3 | |
<!-- /table -->

![seestar express](bench/seestar_express.jpg)

**Base (100 poses)**

<!-- table:base:seestar -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 100 | 93 | |
| Bruit du fond (mag/arcsec² sur 9.5″, plus haut = moins de bruit) | 23.15 | 23.20 | égal |
| Bruit du fond à plus grande échelle (38″) | 24.37 | 24.24 | **nous** |
| Profondeur : étoile détectable à 5σ (magnitude G) | 16.68 | 16.57 | **nous** |
| Étoiles faibles trouvées à 5σ (sur 4769 étoiles Gaia isolées) | 1073 | 1056 | égal |
| Magnitude où la moitié des étoiles est trouvée | 16.30 | 16.31 | égal |
| Rapport signal/bruit sur la cible | 36.4 | 37.5 | égal |
| Finesse des étoiles, FWHM au centre (″) | 5.69 | 5.72 | égal |
| FWHM au bord du champ (″) | 5.69 | 5.77 | égal |
| Rondeur au centre (1 = rond) | 0.86 | 0.91 | Siril |
| Rondeur au bord | 0.89 | 0.88 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 2 | 4 | égal |
| Taches sombres propres à l'image (par degré²) | 0 | 0 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 21.08 | 21.50 | Siril |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.041 | 0.076 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.074 | 0.228 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.69 | 0.50 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.88 | 0.97 | |
| Cible : signal / bruit en mag/arcsec² (info) | 20.08 / 23.98 | 20.12 / 24.05 | |
| Durée (min) | 2.9 | 0.8 | |
<!-- /table -->

![seestar base](bench/seestar_base.jpg)


## Verdict d'ensemble

<!-- verdict -->
Le but : faire au moins aussi bien que Siril 1.4 de base, sur les mêmes poses. Résumé des cinq cas (Base,
100 poses ; entre parenthèses, Express à 30 poses ; M8 : ses 15 poses) :

| | Heart | Iris | M51 | M8 | Seestar |
|---|---|---|---|---|---|
| Bruit fin (mag/arcsec²) | −0,09 (−0,13) | −0,16 (−0,20) | −0,12 (−0,12) | −0,04 | −0,05 (−0,11) |
| Bruit à grande échelle | +0,07 (−0,05) | −0,18 (−0,14) | −0,04 (−0,02) | −0,02 | +0,13 (−0,11) |
| Profondeur 5σ (mag) | +0,01 (−0,14) | −0,15 (−0,19) | −0,07 (−0,04) | −0,05 | +0,11 (−0,09) |
| Étoiles faibles trouvées | +0 % (−1 %) | −3 % (−4 %) | −3 % (−1 %) | −0 % | +2 % (−3 %) |
| Signal/bruit sur la cible | −3 % (−12 %) | −17 % (−24 %) | −18 % (−15 %) | −6 % | −3 % (−2 %) |
| Finesse des étoiles (FWHM) | −12 % (−8 %) | −1 % (0 %) | 0 % (+1 %) | −2 % | −1 % (+2 %) |
| Rondeur au centre | −0,05 (−0,01) | −0,02 (−0,01) | +0,02 (+0,02) | −0,01 | −0,05 (−0,03) |
| Erreur de couleur des étoiles | 3 fois moindre | 1,4 fois moindre | partagée (bleu−rouge : Siril ; vert−rouge : nous) | 1,5 à 2,5 fois moindre | 2 à 3 fois moindre |
| Points colorés résiduels | égal | égal | moins (1-2 contre 14-33 par degré²) | plus (57 contre 7) | égal |
| Rendu (planche) | meilleur | meilleur | meilleur | égal, avec des cœurs d'étoiles bleu-cyan | moins bon (disque de M31) |

Écarts comptés de notre point de vue : un chiffre négatif en magnitude ou en pourcentage de signal/bruit est une
perte ; pour la FWHM, négatif veut dire des étoiles plus fines chez nous.

**Où nous perdons, et de combien :**
1. **Le bruit sur Iris** : 15 à 20 % de plus, à toutes les échelles. Le signal/bruit sur la nébuleuse y perd 17 à
   24 % et la profondeur 0,15 à 0,2 mag.
   - Nos poids de poses en expliquent environ 8 % : de 0,2 à 2,1 fois la médiane, l'équivalent de 86 poses sur 99.
   - Ces poids n'ont pas rendu les étoiles plus fines.
   - **À essayer en premier** : des poids plus doux, ou sans le terme de finesse.
2. **Le bruit fin, en rouge et en bleu.**
   - À 4 pixels, nos canaux rouge et bleu sont 0,2 à 0,3 mag plus bruités que ceux de Siril ; le vert est égal.
   - À 16 pixels, l'écart disparaît, sauf sur Iris (encore 0,1 mag).
   - Cause : le dématriçage et l'interpolation de Siril lissent ces deux couleurs, notre drizzle ne le fait pas. Ce
     n'est pas de l'information perdue, mais cela se voit à 100 % : un grain coloré dans notre fond. Et cela coûte
     environ 15 % de signal/bruit sur les petites échelles des cibles (M51).
   - **Piste** : réduire le bruit de couleur à petite échelle dans la sortie simple, en gardant la luminance.
3. **M31 au Seestar** : notre fond polynomial s'appuie sur 18 % de l'image et prend le grand disque pour du ciel. Le
   disque et les bandes de poussière disparaissent, le ciel tire au violet. Siril (même degré 2) garde le disque.
   C'est le problème déjà connu de l'objet qui remplit le champ.
4. **Rondeur au centre sur les étoiles très petites**, avec le drizzle à 1,5 : 0,86-0,87 contre 0,91-0,92 (Heart,
   Seestar).
5. **Les étoiles saturées** : un croissant ou un cœur bleu-cyan sur les plus brillantes (M8, Iris en Express). Un
   défaut visible à 100 %, que Siril n'a pas.
6. **Les pixels chauds d'un seul canal** (M8 : 57 par degré² contre 7) : de petits points bleus ou rouges. Siril les
   enlève avec sa correction d'après le dark.
7. **Le temps**, voir plus bas : le réseau domine, et le catalogue des étoiles faibles est téléchargé pour rien en
   « Classic » (45 min sur M8).

**Où nous gagnons :**
- la fidélité des couleurs des étoiles, partout sauf l'indice bleu−rouge de M51 ;
- la finesse des étoiles sur Heart ;
- moins de pixels chauds résiduels (M51) ;
- un fond plus propre sur M51 ;
- le rendu de l'H-alpha (Heart), de la nébuleuse bleue (Iris) et de la galaxie (M51).
<!-- /verdict -->

## Durées

Mesurées sur cette machine, pendant que d'autres travaux tournaient (les deux logiciels dans les mêmes conditions).
Siril calcule sur 8 cœurs ; notre moteur sur la carte cuda:0. Notre temps compte tout le flux jusqu'à l'image
finale ; en Base, les catalogues sont repris de l'Express (`prefetch.py`), sans quoi il faudrait ajouter le temps
de téléchargement de l'Express.

Sans le réseau, notre calcul prend à peu près le temps de Siril : un peu plus en Express, souvent moins en Base
(Heart 3,9 contre 5,6 min, Iris 4,7 contre 6,5, M51 6,2 contre 9,6). Le Seestar, aux petites images, reste plus lent
chez nous (1,7 contre 0,8 min).

<!-- durations -->
| Cas | Niveau | Siril (préparation + fin) | Nous (tout) | dont astrométrie + catalogues (réseau) | Nous sans le réseau |
|---|---|---|---|---|---|
| heart | express | 2.3 min (2.0 + 0.3) | 21.8 min | 18.8 min | 2.9 min |
| heart | base | 5.6 min (5.3 + 0.3) | 9.4 min | 5.5 min | 3.9 min |
| iris | express | 3.2 min (2.8 + 0.4) | 7.9 min | 4.7 min | 3.2 min |
| iris | base | 6.5 min (6.2 + 0.3) | 8.9 min | 4.2 min | 4.7 min |
| m51 | express | 2.9 min (2.4 + 0.6) | 4.0 min | 0.9 min | 3.1 min |
| m51 | base | 9.6 min (9.1 + 0.5) | 8.4 min | 2.2 min | 6.2 min |
| m8 | express | 4.2 min (3.5 + 0.6) | 81.4 min | 75.6 min | 5.8 min |
| seestar | express | 0.3 min (0.2 + 0.1) | 4.0 min | 3.6 min | 0.4 min |
| seestar | base | 0.8 min (0.8 + 0.1) | 2.9 min | 1.3 min | 1.7 min |
<!-- /durations -->

## Problèmes trouvés dans le moteur

Rien n'a été modifié dans le moteur ; voici ce que le banc a montré, avec l'endroit dans le code.

1. **Le cache des catalogues ignore ASTRO_GPU_HOME.** `stages/catalogs.py` (ligne 16) et `core/blindsolve.py`
   (ligne 18) écrivent dans `Path.home()/.astro_gpu/cache`, quel que soit ASTRO_GPU_HOME. Un essai ou un banc écrit donc
   dans le dossier de l'utilisateur. Le banc l'évite en déplaçant USERPROFILE pour ses calculs.
2. **Le cache des catalogues ne resservira presque jamais.** La clé d'un cône Gaia est son centre exact (5 décimales,
   `_query_cells`), qui suit la solution astrométrique de la pile. Une autre pile du même champ (d'autres poses, une
   autre pose de référence) ne retrouve rien et retélécharge tout : 15 min pour Heart (54 cellules) avec VizieR en
   panne. Le banc contourne avec `tools/bench/prefetch.py`, qui découpe les cônes de Base dans ceux d'Express. Piste :
   un cache par tuiles fixes du ciel (HEALPix), indépendant de la solution.
3. **Quand VizieR ne répond pas (il se tait, sans refuser), chaque calcul attend tout son délai avant de passer aux
   secours.** `core/gaia_archive.py`, `gaia_cone` : la première requête VizieR d'un processus attend 600 s
   (astrométrie) ou 900 s (catalogues, et chacun des 6 fils en parallèle part avant que le drapeau « VizieR en panne »
   ne soit levé). Sur M8, l'astrométrie a pris 10 min au lieu d'une. Piste : un délai court (20-30 s) pour savoir si
   VizieR répond, puis le délai long seulement pour les grosses requêtes.
4. **L'astrométrie n'a aucun cache disque.** `stages/astrometry.py`, `gaia_bright` : le catalogue des étoiles
   brillantes reste en mémoire. Chaque nouvelle pile demande 2 à 4 min de réseau et échoue hors ligne.
5. **« Classic stack » télécharge un catalogue dont il ne se sert pas.** L'étape catalogues charge toujours les étoiles
   faibles (G 18,5 à 20,5, `gaia_faint.npz`), que seuls les blocs Étoiles et Fond « H-alpha » lisent
   (`stages/stars.py`, `stages/background.py`). Classic n'utilise que les spectres XP. Sur le champ très dense de M8,
   ce téléchargement inutile dure plus longtemps que tout le reste du calcul. Piste : ne demander les étoiles faibles
   que si un bloc de la chaîne les lit (une étape « catalogue faible » à part, partagée par les flux Full).
6. **Mémoire à la calibration.** Sur M8 (26 Mpx), la première tentative a échoué : « MemoryError » dans un des
   8 processus de calibration, pendant que Siril empilait à côté. La seconde a réussi. Piste : régler le nombre de
   processus sur la mémoire libre.

Côté image, à regarder en priorité (constats du banc, sans correction) :

7. **Poids des poses (Iris)** : de 0,2 à 2,1 fois la médiane, soit 86 poses utiles sur 99, sans étoiles plus fines.
   Notre bruit y dépasse celui de Siril de 15 à 20 %. Voir `stages/stack.py`, poids (transparence/bruit)² × finesse.
8. **Cœurs des étoiles saturées** : un croissant ou un point bleu-cyan sur les étoiles les plus brillantes (M8 ; Iris
   en Express) dans la sortie simple. Siril les rend blanches.
9. **Pixels chauds d'un seul canal** : M8 garde 57 petits points bleus ou rouges par degré² (Siril : 7). Une
   correction des pixels chauds d'après le master dark, comme le `-cc=dark` de Siril, les enlèverait.
10. **Bruit de couleur à petite échelle** : sans interpolation, nos canaux rouge et bleu gardent un grain fin, de 0,2 à
    0,3 mag plus fort qu'avec Siril à 4 pixels. Il disparaît à 16 pixels.
11. **Fond du Seestar M31** : le polynôme s'appuie sur 18 % de l'image (`sky_fit_blocks_pct`) et prend le disque pour
    du ciel.
12. **Rondeur au centre avec le drizzle 1,5** sur des étoiles de 2 à 2,5 pixels (Heart, Seestar) : 0,86-0,87 contre
    0,91-0,92.


## Limites du banc

- **Un seul passage par cas**, sur une machine chargée par d'autres travaux. Un écart de 0,05 mag ou de 3 % est sous
  le seuil de la mesure ; il est noté « égal ».
- **Le bruit dépend de l'échelle.** L'interpolation de Siril (dématriçage, puis Lanczos à l'alignement) lisse le bruit
  d'un pixel à l'autre sans ajouter d'information. Notre drizzle garde chaque pixel brut. D'où deux échelles mesurées :
  la grande est la plus juste pour comparer l'information.
- **La « cible »** est la région étendue la plus brillante, prise automatiquement sur l'image de Siril : le cœur des
  nébuleuses, le noyau et les bras de M51, le bulbe de M31. Elle ne dit rien du disque faible de M31, que seule la
  planche montre.
- **La planéité** est faussée dans les champs pleins de nébulosité (Heart, Iris, M8) : la nébulosité y compte comme un
  écart.
- **Les couleurs** sont jugées sur les étoiles, pas sur les nébuleuses. La référence de blanc diffère : la galaxie
  spirale moyenne pour la SPCC de Siril, sa propre référence pour la PCC, le Soleil pour nous. Le « point blanc » des
  tableaux est donc une information, pas une note.
- **Le rendu final** n'est pas comparable en goût : Siril éclaire le fond, nous le gardons sombre. Les planches
  servent à voir les défauts.

## Refaire le banc

Python : `C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe`, depuis `C:/Users/Nicolas/sandbox/astro`.

```
python tools/bench/build.py                          # copies figées, manifestes, état du banc (une fois)
python tools/bench/siril/install_catalogues.py       # catalogues locaux de Siril (une fois)
python tools/bench/siril/run_siril.py CAS NIVEAU     # Siril (8 cœurs)
python tools/bench/ours.py CAS express               # notre moteur (cuda:0)
python tools/bench/ours.py CAS base --stop-before catalogs
python tools/bench/prefetch.py CAS                   # catalogues de Base repris de l'Express
python tools/bench/ours.py CAS base
python tools/bench/measure.py CAS NIVEAU             # mesures + planche
python tools/bench/report.py fill                    # tableaux et durées de ce document, planches copiées
```

Après un changement du moteur, relancer `ours.py` suffit : seuls les blocs touchés sont recalculés.
Tout ce qui est produit est sous `F:/astro_essais/bench/`.
