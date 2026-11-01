# Banc de qualité : notre moteur face à Siril 1.4

*Passe finale, 2026-11-01 : le moteur du dépôt avec les 8 correctifs du chemin de base (commit 8d450c1), six cas,
Express et Base. Les mesures, les tableaux et les planches se refont avec `tools/bench/` (voir la fin).*

## En bref

<!-- enbref -->
Sur les mêmes fichiers, notre « Classic stack » fait **jeu égal avec Siril 1.4 de base sur la profondeur et les étoiles
faibles en Base** (100 poses) sur Heart, M51, Rosette et le Seestar (de −0,07 à +0,17 mag), **passe devant sur M8**
(le jeu du tutoriel Siril : +0,19 mag, étoiles 5 % plus fines), avec des **couleurs
d'étoiles plus fidèles** partout (1,2 à 3 fois moins d'erreur face à Gaia, sauf le bleu−rouge de M51), un **fond plus
plat** sur Heart, Iris et M51, des étoiles plus fines sur Heart et **un rendu plus coloré et plus lisible** sur chaque
planche.

Ce qui reste moins bien :
- **Iris** : 0,11 à 0,17 mag de bruit en plus, −12 à −20 % de signal/bruit sur la nébuleuse ;
- **le bruit fin** (à 4 pixels) : 0,03 à 0,13 mag derrière Siril partout ;
- **des petits points d'une seule couleur** : 14 à 191 par degré² contre 0 à 6 chez Siril (nouvelle mesure ; hors M8, où elle se noie dans la nébuleuse) ; sur les
  poses de la Rosette, le moteur corrigé en laisse plus que celui d'avant (77 contre 23) : à regarder en premier ;
- **en Express quand le ciel est nuageux** (Rosette), nos poses écartées coûtent 0,25 mag de profondeur ;
- le disque de M31 au Seestar, visible mais plus pâle que chez Siril.

Aucune image de calibration des six cas n'est mauvaise : seules les 100 fausses darks de la Rosette l'étaient, et le
banc emploie les darks de la nuit d'Iris, qui vont bien avec ses lights.
<!-- /enbref -->

## Le banc

Six cas, sans recadrage, avec exactement les mêmes fichiers pour les deux logiciels (des copies figées sous
`F:/astro_essais/bench/<cas>/data/`, la liste dans `manifest.json`). Nos projets sont `ours_express` et `ours_base`
de chaque cas : des projets v2 que le client ouvre (Accueil, section « Bench against Siril »).

| Cas | Appareil et instrument | Express | Base | Calibration |
|---|---|---|---|---|
| heart | Sony A7S défiltré, lunette TS61 (272 mm) | 30 poses de 721 | 100 | 28 darks, 46 flats, pas d'offsets |
| iris | Sony A7S défiltré, Newton 130/900 | 30 de 1268 | 100 | 48 darks, 62 flats, pas d'offsets |
| m51 | Canon 6D, Newton 200/1000 | 30 de 159 | 100 | 25 darks, 12 flats, 10 offsets |
| m8 | ZWO ASI2600MC (le jeu du tutoriel Siril) | 15 de 15 | = Express | 15 darks, 15 flats, 15 offsets |
| seestar | Seestar S50, M31 | 30 de 360 | 100 | aucune (le Seestar n'en fait pas) |
| rosette | Sony A7S défiltré, Newton 130/900, ISO 4000, 15 s | 30 de 358 | 100 | les 48 darks de la nuit d'Iris, les 62 flats d'Iris |

Les poses sont prises à intervalles réguliers sur toute la séance. Pour M8, seules les 15 poses à large bande sont
prises (les 20 poses bi-bande ont un autre filtre et d'autres flats) : son niveau Base est son Express. Pour la
Rosette, les « darks » de la nuit sont des photos du ciel (`DARKS_28_12_24.md`) : le banc prend la correction choisie
par l'utilisateur, les 48 vraies darks de la nuit d'Iris (même boîtier, 15 s, ISO 4000).

### Les images de calibration de chaque cas

Le contrôle du moteur (`core/diagnostics.py`, `api.calibration_frames`) juge chaque dark, flat et offset seule ; une
image « mauvaise » n'est pas de son type (une dark avec des étoiles, par exemple). Il a été passé sur tous les projets
du banc, Express et Base (`tools/bench/calcheck.py`, `bench/calcheck.json`) :

<!-- calcheck -->
| Cas | Darks | Flats | Offsets | Verdict |
|---|---|---|---|---|
| heart | 28 : 28 bonnes | 46 : 46 bonnes | – | aucune image mauvaise |
| iris | 48 : 48 bonnes | 62 : 58 bonnes, 4 à vérifier | – | aucune image mauvaise |
| m51 | 25 : 16 bonnes, 9 à vérifier | 12 : 2 bonnes, 10 à vérifier | 10 : 10 bonnes | aucune image mauvaise |
| m8 | 15 : 15 bonnes | 15 : 15 bonnes | 15 : 15 bonnes | aucune image mauvaise |
| seestar | – | – | – | pas d'images de calibration |
| rosette | 48 : 48 bonnes | 62 : 58 bonnes, 4 à vérifier | – | aucune image mauvaise |
<!-- /calcheck -->

<!-- calcheck:verdict -->
**Aucun cas n'avait d'image de calibration mauvaise ; aucun n'est donc à refaire pour cela.** Seules les 100 fausses
darks de la Rosette étaient mauvaises (de 26 à 410 étoiles chacune) : le banc ne les emploie pas, il prend les darks
d'Iris, toutes bonnes. Les « à vérifier » ne changent rien au calcul :
- M51 : 9 darks 5 ADU plus hautes que les 16 autres (le capteur s'est un peu réchauffé), et 10 flats sur 12 trop peu
  exposés (5 à 9 % de la pleine échelle, quand 30 à 60 % est conseillé) ;
- Iris et Rosette (les mêmes flats) : 4 flats à 14 % au lieu de 12-13 % (la lumière a un peu changé).
<!-- /calcheck:verdict -->

## La recette Siril

Siril 1.4.0-beta3 en ligne de commande (`siril-cli`), le script officiel « OSC_Preprocessing » de Siril 1.4, puis le
traitement de base qu'un utilisateur de Siril enchaîne. Pas de GraXpert, pas d'IA, pas de retrait d'étoiles. Les
scripts sont gardés à côté des résultats (`siril_<niveau>/pre.ssf` et `post.ssf`).

1. **Conversion** des fichiers bruts (RAW ou FITS) en séquences Siril.
2. **Maître d'offset** : moyenne des offsets avec rejet winsorisé 3/3, sans normalisation. Sans offsets (les cas
   Sony A7S) : un offset synthétique au niveau de noir du capteur (512), comme Siril le conseille pour les APN.
3. **Maître de flat** : chaque flat moins l'offset, puis moyenne winsorisée 3/3 avec normalisation multiplicative.
4. **Maître de dark** : moyenne winsorisée 3/3 sans normalisation. Pas d'optimisation de la dark (`-opt`) : le script
   officiel ne l'emploie pas (Rosette comprise, où les darks viennent d'une autre nuit).
5. **Calibration des poses** : moins le dark, divisées par le flat (couleurs du flat égalisées), pixels chauds
   corrigés d'après le dark (3 sigma), puis dématriçage (RCD, le réglage par défaut).
6. **Alignement** global sur les étoiles (homographie, interpolation Lanczos, les réglages par défaut).
7. **Empilement** : moyenne avec rejet winsorisé 3/3, normalisation additive avec mise à l'échelle, fonds RVB égalisés,
   résultat en 32 bits. Toutes les poses sont gardées. Pour le Seestar, le script Seestar officiel de Siril :
   alignement en deux passes, et les poses aux étoiles les moins rondes écartées.
8. **Astrométrie** par Siril, avec le centre approximatif et la focale (les mêmes que ceux donnés à notre moteur).
9. **Extraction du fond** : polynôme de degré 2 (le même degré que notre bloc Dégradé), 20 échantillons par ligne,
   tolérance 1.
10. **Couleur** : SPCC quand le capteur est dans la base de Siril (ASI2600MC, Seestar S50, filtre UV/IR) ; PCC sinon
    (Sony A7S défiltré, Canon 6D).
11. **Étirement automatique lié**, enregistré en TIFF 16 bits et en PNG.

Deux écarts au « tout par défaut » : les catalogues Gaia DR3 locaux de Siril (VizieR ne répondait pas), installés dans
`F:/astro_essais/bench/_siril/catalogues` ; et une copie des réglages de l'utilisateur (`_siril/config.ini`), avec
8 cœurs sur 16.

## Nos réglages

Le flux « Classic stack » du moteur, réglages suggérés, sans rien toucher : calibration, alignement, empilement,
astrométrie, couleur, dégradé polynomial (degré 2), sortie simple. Le projet reçoit le même centre approximatif et la
même focale que Siril. Calcul sur la carte cuda:0 ; l'état du moteur (profils d'appareil, cache des catalogues) est
dans `F:/astro_essais/bench/_home`, et il lit d'abord les catalogues locaux de Siril
(`ASTRO_GPU_SIRIL_CATALOGUES`). Le moteur choisit lui-même le drizzle (1 à 30 poses, 1,5 à 100 poses pour les
petites étoiles) et peut écarter des poses ; Siril les garde toutes.

## Les mesures

Chaque image garde sa propre grille : tout est placé sur le ciel grâce à l'astrométrie de chaque logiciel, et les
tailles sont en secondes d'arc. Seul le ciel couvert par les deux images compte. Les étoiles de référence sont celles
de Gaia DR3. Chaque couleur des deux piles est d'abord ramenée à la même échelle photométrique (une étoile de
magnitude G donne le même flux partout).

Sur les **piles linéaires** (notre pile brute, la pile de Siril avant extraction du fond) :
- **bruit du fond** : l'écart des moyennes de petites boîtes de ciel (étoiles masquées), en magnitude par seconde
  d'arc² (plus haut = moins de bruit), à deux échelles : boîtes de 4 pixels de Siril, et 4 fois plus grandes ;
- **profondeur** : la magnitude de l'étoile qu'on détecte à 5 sigma ;
- **étoiles faibles** : combien d'étoiles Gaia isolées ressortent à plus de 5 sigma, et la magnitude où on en trouve la
  moitié ;
- **signal/bruit sur la cible** : la région étendue la plus brillante contre le ciel juste autour ;
- **étoiles** : finesse (largeur à mi-hauteur) et rondeur, au centre et au bord du champ ;
- **défauts** : traînées longues restantes ; taches claires ou sombres propres à une image ; **points d'une seule
  couleur** (au-dessus de 6σ dans une couleur, sous 2σ dans les deux autres : les pixels chauds restants) ;
  **traînées le long de la dérive** (le « walking noise ») : le fond est moyenné le long de lignes de 160″ dans toutes
  les directions, et le bruit des lignes posées dans le sens de la dérive de la monture (connue par notre recalage)
  est comparé au bruit moyen des autres : 1 = pas de traînées.

Sur les **images linéaires finales** (fond retiré, couleurs étalonnées) :
- **planéité du fond** : l'écart du « plancher » de 8×8 zones, en magnitude par seconde d'arc² (plus haut = plus
  plat ; dans un champ plein de nébulosité, une partie de l'écart est la nébulosité) ;
- **couleurs des étoiles** : l'indice bleu−rouge (et vert−rouge) de chaque étoile face à sa couleur Gaia BP−RP ;
  l'erreur est la dispersion autour d'une droite, en unités de BP−RP (plus bas = plus fidèle).

Une planche par cas et par niveau : Siril à gauche, nous à droite, la même zone du ciel, et quatre recadrages à 100 %.
Les deux rendus par défaut ne visent pas le même fond (Siril l'éclaire, nous le gardons sombre) : la planche sert à
juger les défauts, pas le goût.

## Verdict d'ensemble

<!-- verdict -->
Le but : faire au moins aussi bien que Siril 1.4 de base, sur les mêmes poses.
- **Atteint en Base sur Heart, M51, Rosette et le Seestar** pour la profondeur et les étoiles faibles (écarts de
  −0,07 à +0,17 mag, −3 à +3 %), avec de meilleures couleurs et, sauf le Seestar et la Rosette, un fond plus plat.
- **Pas encore sur Iris** (−0,14 mag, −12 % de signal/bruit en Base), ni sur le bruit fin et les points d'une seule
  couleur, partout.
- **En Express**, Siril garde un petit avantage de profondeur (0,07 à 0,13 mag ; 0,25 sur la Rosette, où nous avons
  écarté 14 poses nuageuses sur 30).
- **M8** (le jeu du tutoriel Siril) : devant Siril (profondeur +0,19 mag, étoiles 5 % plus fines, fond plus plat).
- Depuis le banc du matin, les correctifs ont gagné : M51 −16 % → −9 % de signal/bruit, Iris −23 % → −20 % (Express),
  le fond de Heart et de M8 plus plat que celui de Siril, plus de croissant bleu-cyan sur les étoiles saturées, le
  ciel du Seestar neutre.
<!-- /verdict -->

Notre écart à Siril, cas par cas (Base, et Express entre parenthèses ; M8 : ses 15 poses). Magnitudes et pourcentages
positifs = mieux chez nous ; FWHM négative = étoiles plus fines chez nous.

<!-- overall -->
| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Bruit fin (mag/arcsec²) | −0,08 (−0,10) | −0,12 (−0,13) | −0,07 (−0,08) | −0,03 | −0,03 (−0,10) | −0,12 (−0,11) |
| Bruit à grande échelle | +0,05 (−0,05) | −0,17 (−0,11) | −0,03 (−0,02) | −0,02 | +0,08 (−0,12) | −0,12 (−0,07) |
| Profondeur 5σ (mag) | +0,00 (−0,13) | −0,14 (−0,11) | −0,07 (−0,07) | +0,19 | +0,17 (−0,08) | −0,04 (−0,24) |
| Étoiles faibles trouvées | +1 % (−1 %) | −2 % (−1 %) | −3 % (−2 %) | +3 % | +3 % (−2 %) | −0 % (−3 %) |
| Signal/bruit sur la cible | −0 % (−9 %) | −12 % (−20 %) | −9 % (−8 %) | −5 % | −4 % (+4 %) | −7 % (−11 %) |
| Finesse des étoiles (FWHM au centre) | −12 % (−8 %) | −2 % (+1 %) | +0 % (+1 %) | −5 % | −0 % (+2 %) | +2 % (+2 %) |
| Rondeur au centre | −0,04 (−0,00) | −0,01 (−0,01) | +0,01 (+0,01) | +0,01 | −0,05 (−0,03) | −0,01 (−0,00) |
| Planéité du fond (mag/arcsec²) | +0,08 (+0,24) | +0,14 (+0,39) | +0,18 (+0,24) | +0,15 | −0,15 (−0,21) | −0,10 (−0,02) |
| Erreur de couleur B−R (nous / Siril) | 0,069 / 0,174 (0,097 / 0,218) | 0,038 / 0,049 (0,048 / 0,058) | 0,026 / 0,018 (0,026 / 0,021) | 0,061 / 0,073 | 0,063 / 0,076 (0,059 / 0,152) | 0,034 / 0,041 (0,042 / 0,053) |
| Erreur de couleur V−R (nous / Siril) | 0,111 / 0,254 (0,167 / 0,326) | 0,041 / 0,052 (0,061 / 0,070) | 0,028 / 0,040 (0,037 / 0,047) | 0,051 / 0,128 | 0,069 / 0,228 (0,191 / 0,262) | 0,052 / 0,089 (0,078 / 0,103) |
| Points d'une seule couleur par degré² (nous / Siril) | 191 / 1 (22 / 1) | 89 / 3 (76 / 3) | 18 / 5 (24 / 4) | 2690 / 1414 | 25 / 0 (14 / 0) | 76 / 4 (77 / 6) |
| Traînées le long de la dérive (nous / Siril ; 1 = aucune) | 1,00 / 1,00 (1,00 / 1,00) | 1,00 / 1,01 (0,99 / 1,00) | 1,04 / 1,06 (1,04 / 1,05) | 0,99 / 1,00 | 1,05 / 1,05 (0,98 / 0,97) | 1,04 / 1,03 (1,00 / 1,00) |
<!-- /overall -->

## Résultats par cas

Chaque tableau : notre valeur, celle de Siril, et qui fait mieux (« égal » sous le seuil de la mesure : 0,05 mag,
3 à 5 %). Sous chaque tableau, la planche (Siril à gauche).

### Heart (Sony A7S défiltré, lunette 272 mm)

<!-- verdict:heart -->
**À égalité en Base, devant Siril sur la finesse, les couleurs et le fond ; un peu derrière en Express.**
- **Base** : profondeur, étoiles faibles et signal/bruit sur le cœur de la nébuleuse égaux (16,62 contre 16,62 ;
  6,4 contre 6,4) ; bruit plus faible à grande échelle (+0,05 mag), un peu plus fort à petite (−0,08).
- **Express** : Siril va 0,13 mag plus profond, avec 0,10 mag de bruit fin en moins ; signal/bruit −8 %.
- **Étoiles 8 à 12 % plus fines chez nous** (11,7″ contre 13,3″ au centre en Base) et plus rondes au bord ; un peu
  moins rondes au centre avec le drizzle 1,5 (0,88 contre 0,92).
- **Couleurs des étoiles** 2,3 à 2,5 fois plus fidèles ; **fond** plus plat en Express (+0,23 mag), égal en Base.
- **Planche** : notre H-alpha est rouge franc, celui de Siril rose pâle sur un fond gris.
- **Points d'une seule couleur** : 22 par degré² en Express, **191 en Base** (Siril : 1). C'est le chiffre le plus
  fort du banc, avec le drizzle 1,5 (voir « Ce qui reste »).
<!-- /verdict -->

**Express (30 poses)**

<!-- table:express:heart -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 30 | 30 | |
| Bruit du fond (mag/arcsec² sur 25.5″, plus haut = moins de bruit) | 24.50 | 24.60 | Siril |
| Bruit du fond à plus grande échelle (102″) | 25.24 | 25.29 | égal |
| Profondeur : étoile détectable à 5σ (magnitude G) | 16.24 | 16.37 | Siril |
| Étoiles faibles trouvées à 5σ (sur 25919 étoiles Gaia isolées) | 22753 | 22992 | égal |
| Magnitude où la moitié des étoiles est trouvée | 15.69 | 15.75 | Siril |
| Rapport signal/bruit sur la cible | 4.6 | 5.0 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 12.07 | 13.10 | **nous** |
| FWHM au bord du champ (″) | 14.64 | 15.06 | égal |
| Rondeur au centre (1 = rond) | 0.91 | 0.91 | égal |
| Rondeur au bord | 0.83 | 0.78 | **nous** |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 0 | 0 | égal |
| Taches sombres propres à l'image (par degré²) | 0 | 0 | égal |
| Points d'une seule couleur : pixels chauds restants (par degré²) | 22 | 1 | Siril |
| Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune) | 1.00 | 1.00 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 24.61 | 24.38 | **nous** |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.097 | 0.218 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.167 | 0.326 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.37 | 0.41 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.88 | 0.93 | |
| Cible : signal / bruit en mag/arcsec² (info) | 23.48 / 25.13 | 23.53 / 25.29 | |
| Durée (min) | 4.1 | 2.3 | |
<!-- /table -->

![heart express](bench/heart_express.jpg)

**Base (100 poses)**

<!-- table:base:heart -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 99 | 100 | |
| Bruit du fond (mag/arcsec² sur 25.5″, plus haut = moins de bruit) | 24.74 | 24.82 | Siril |
| Bruit du fond à plus grande échelle (102″) | 25.33 | 25.28 | **nous** |
| Profondeur : étoile détectable à 5σ (magnitude G) | 16.62 | 16.62 | égal |
| Étoiles faibles trouvées à 5σ (sur 26716 étoiles Gaia isolées) | 24654 | 24474 | égal |
| Magnitude où la moitié des étoiles est trouvée | 16.00 | 15.97 | égal |
| Rapport signal/bruit sur la cible | 6.4 | 6.4 | égal |
| Finesse des étoiles, FWHM au centre (″) | 11.65 | 13.30 | **nous** |
| FWHM au bord du champ (″) | 14.35 | 15.26 | **nous** |
| Rondeur au centre (1 = rond) | 0.88 | 0.92 | Siril |
| Rondeur au bord | 0.84 | 0.77 | **nous** |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 0 | 0 | égal |
| Taches sombres propres à l'image (par degré²) | 0 | 0 | égal |
| Points d'une seule couleur : pixels chauds restants (par degré²) | 191 | 1 | Siril |
| Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune) | 1.00 | 1.00 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 24.78 | 24.70 | égal |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.069 | 0.174 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.111 | 0.254 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.40 | 0.43 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.91 | 0.91 | |
| Cible : signal / bruit en mag/arcsec² (info) | 23.41 / 25.43 | 23.53 / 25.55 | |
| Durée (min) | 5.6 | 5.6 | |
<!-- /table -->

![heart base](bench/heart_base.jpg)

### Iris (Sony A7S défiltré, Newton 130/900)

<!-- verdict:iris -->
**Siril gagne encore sur le bruit, nous sur les couleurs et le fond.** C'est le cas où l'écart reste le plus net.
- **Bruit** : 0,11 à 0,17 mag de plus chez nous, à toutes les échelles ; Siril va 0,11 à 0,14 mag plus profond. Le
  nombre d'étoiles faibles trouvées est presque le même (−1 à −2 %).
- **Signal/bruit sur la nébuleuse** : −12 % en Base, −20 % en Express, à signal égal.
- **Étoiles** : finesse et rondeur égales. Les 30 et 99 poses gardées (Siril : 30 et 100).
- **Couleurs des étoiles** plus fidèles chez nous (0,038 contre 0,049 en B−R) ; **fond** plus plat (+0,14 à +0,39 mag).
- **Planche** : notre nébuleuse est bleue et structurée, celle de Siril blanc-gris.
- **Points d'une seule couleur** : 76 à 89 par degré² (Siril : 3).
<!-- /verdict -->

**Express (30 poses)**

<!-- table:express:iris -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 30 | 30 | |
| Bruit du fond (mag/arcsec² sur 7.7″, plus haut = moins de bruit) | 23.72 | 23.85 | Siril |
| Bruit du fond à plus grande échelle (31″) | 25.11 | 25.22 | Siril |
| Profondeur : étoile détectable à 5σ (magnitude G) | 17.33 | 17.44 | Siril |
| Étoiles faibles trouvées à 5σ (sur 14441 étoiles Gaia isolées) | 4922 | 4951 | égal |
| Magnitude où la moitié des étoiles est trouvée | 16.84 | 16.87 | égal |
| Rapport signal/bruit sur la cible | 2.4 | 3.0 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 6.15 | 6.12 | égal |
| FWHM au bord du champ (″) | 7.79 | 7.95 | égal |
| Rondeur au centre (1 = rond) | 0.88 | 0.89 | égal |
| Rondeur au bord | 0.65 | 0.64 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 15 | 4 | Siril |
| Taches sombres propres à l'image (par degré²) | 5 | 3 | égal |
| Points d'une seule couleur : pixels chauds restants (par degré²) | 76 | 3 | Siril |
| Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune) | 0.99 | 1.00 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 23.33 | 22.94 | **nous** |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.048 | 0.058 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.061 | 0.070 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.51 | 0.67 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.90 | 0.87 | |
| Cible : signal / bruit en mag/arcsec² (info) | 23.50 / 24.45 | 23.40 / 24.59 | |
| Durée (min) | 3.7 | 3.2 | |
<!-- /table -->

![iris express](bench/iris_express.jpg)

**Base (100 poses)**

<!-- table:base:iris -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 99 | 100 | |
| Bruit du fond (mag/arcsec² sur 7.7″, plus haut = moins de bruit) | 24.42 | 24.54 | Siril |
| Bruit du fond à plus grande échelle (31″) | 25.68 | 25.85 | Siril |
| Profondeur : étoile détectable à 5σ (magnitude G) | 17.96 | 18.10 | Siril |
| Étoiles faibles trouvées à 5σ (sur 15647 étoiles Gaia isolées) | 6697 | 6833 | égal |
| Magnitude où la moitié des étoiles est trouvée | 17.48 | 17.51 | égal |
| Rapport signal/bruit sur la cible | 3.7 | 4.2 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 5.96 | 6.05 | égal |
| FWHM au bord du champ (″) | 7.38 | 7.54 | égal |
| Rondeur au centre (1 = rond) | 0.89 | 0.90 | égal |
| Rondeur au bord | 0.65 | 0.66 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 15 | 5 | Siril |
| Taches sombres propres à l'image (par degré²) | 9 | 4 | Siril |
| Points d'une seule couleur : pixels chauds restants (par degré²) | 89 | 3 | Siril |
| Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune) | 1.00 | 1.01 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 23.54 | 23.40 | **nous** |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.038 | 0.049 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.041 | 0.052 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.50 | 0.67 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.90 | 0.87 | |
| Cible : signal / bruit en mag/arcsec² (info) | 23.74 / 25.17 | 23.75 / 25.31 | |
| Durée (min) | 7.0 | 6.5 | |
<!-- /table -->

![iris base](bench/iris_base.jpg)

### M51 (Canon 6D, Newton 200/1000)

<!-- verdict:m51 -->
**Presque à égalité, avec un fond plus propre et un rendu meilleur chez nous.**
- **Profondeur** −0,06 mag, étoiles faibles −2 à −3 %, bruit fin −0,07 à −0,08 mag ; à grande échelle, égal.
- **Signal/bruit sur la galaxie** : −7 % (Express) et −9 % (Base), à signal égal (il était de −15 à −18 % au matin).
- **Étoiles** : finesse et rondeur égales.
- **Fond** plus plat chez nous (+0,18 à +0,25 mag) : Siril laisse une zone bleutée sur la moitié droite. Siril
  laisse aussi plus de taches claires propres à l'image (11 contre 5 par degré² en Express).
- **Couleurs** : bleu−rouge un peu plus juste chez Siril (0,018 contre 0,026), vert−rouge chez nous (0,028 contre
  0,040). Avec sa PCC, Siril rend blanche une étoile de BP−RP 0,4 : les étoiles de type solaire sortent orangées.
- **Planche** : nos bras sont bleus, avec un cœur jaune et des régions HII ; la galaxie de Siril est grise.
- **Points d'une seule couleur** : 18 à 24 par degré² (Siril : 4 à 5).
<!-- /verdict -->

**Express (30 poses)**

<!-- table:express:m51 -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 30 | 30 | |
| Bruit du fond (mag/arcsec² sur 5.4″, plus haut = moins de bruit) | 24.58 | 24.66 | Siril |
| Bruit du fond à plus grande échelle (22″) | 25.89 | 25.91 | égal |
| Profondeur : étoile détectable à 5σ (magnitude G) | 18.20 | 18.26 | Siril |
| Étoiles faibles trouvées à 5σ (sur 3602 étoiles Gaia isolées) | 1254 | 1281 | égal |
| Magnitude où la moitié des étoiles est trouvée | 17.57 | 17.64 | Siril |
| Rapport signal/bruit sur la cible | 4.3 | 4.6 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 6.96 | 6.88 | égal |
| FWHM au bord du champ (″) | 8.41 | 8.36 | égal |
| Rondeur au centre (1 = rond) | 0.83 | 0.82 | égal |
| Rondeur au bord | 0.65 | 0.65 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 5 | 11 | **nous** |
| Taches sombres propres à l'image (par degré²) | 2 | 1 | égal |
| Points d'une seule couleur : pixels chauds restants (par degré²) | 24 | 4 | Siril |
| Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune) | 1.04 | 1.05 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 23.19 | 22.94 | **nous** |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.026 | 0.021 | égal |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.037 | 0.047 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.45 | 0.80 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.88 | 0.42 | |
| Cible : signal / bruit en mag/arcsec² (info) | 23.51 / 25.09 | 23.50 / 25.17 | |
| Durée (min) | 4.6 | 2.9 | |
<!-- /table -->

![m51 express](bench/m51_express.jpg)

**Base (100 poses)**

<!-- table:base:m51 -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 100 | 100 | |
| Bruit du fond (mag/arcsec² sur 5.4″, plus haut = moins de bruit) | 25.23 | 25.30 | Siril |
| Bruit du fond à plus grande échelle (22″) | 26.45 | 26.49 | égal |
| Profondeur : étoile détectable à 5σ (magnitude G) | 18.72 | 18.78 | Siril |
| Étoiles faibles trouvées à 5σ (sur 3784 étoiles Gaia isolées) | 1577 | 1627 | Siril |
| Magnitude où la moitié des étoiles est trouvée | 18.01 | 18.07 | Siril |
| Rapport signal/bruit sur la cible | 3.1 | 3.4 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 7.06 | 7.03 | égal |
| FWHM au bord du champ (″) | 8.49 | 8.47 | égal |
| Rondeur au centre (1 = rond) | 0.83 | 0.81 | égal |
| Rondeur au bord | 0.66 | 0.66 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 2 | 29 | **nous** |
| Taches sombres propres à l'image (par degré²) | 2 | 1 | égal |
| Points d'une seule couleur : pixels chauds restants (par degré²) | 18 | 5 | Siril |
| Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune) | 1.04 | 1.06 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 23.23 | 23.05 | **nous** |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.026 | 0.018 | Siril |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.028 | 0.040 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.46 | 0.80 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.88 | 0.35 | |
| Cible : signal / bruit en mag/arcsec² (info) | 24.44 / 25.68 | 24.41 / 25.75 | |
| Durée (min) | 8.4 | 9.6 | |
<!-- /table -->

![m51 base](bench/m51_base.jpg)

### M8 et M20 (ASI2600MC, le jeu du tutoriel Siril)

<!-- verdict:m8 -->
**Nous passons devant Siril sur son propre jeu de tutoriel.**
- **Profondeur +0,19 mag**, 3 % d'étoiles faibles en plus, magnitude à 50 % +0,15 ; bruit égal aux deux échelles.
- **Étoiles 5 % plus fines** (5,33″ contre 5,63″ au centre), aussi rondes.
- **Fond plus plat** (+0,15 mag) ; **couleurs des étoiles** plus fidèles (V−R 0,051 contre 0,128), bien que Siril ait
  ici sa SPCC avec le bon capteur.
- **Signal/bruit sur le cœur de la Lagune** : −6 %.
- Les défauts du matin ont disparu : plus de croissant bleu-cyan sur les étoiles saturées ; 9 taches claires propres à
  l'image par degré² contre 11 chez Siril (57 au matin). Le compte de « points d'une seule couleur » ne veut rien dire
  ici (2 690 contre 1 414 chez Siril) : le champ est rempli de nébulosité rouge et d'étoiles faibles.
- **Temps** : 29 min, dont l'essentiel à attendre le catalogue des étoiles faibles de ce champ très dense (GAVO).
<!-- /verdict -->

**Express = Base (15 poses)**

<!-- table:express:m8 -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 15 | 15 | |
| Bruit du fond (mag/arcsec² sur 8.0″, plus haut = moins de bruit) | 24.05 | 24.09 | égal |
| Bruit du fond à plus grande échelle (32″) | 24.36 | 24.38 | égal |
| Profondeur : étoile détectable à 5σ (magnitude G) | 17.90 | 17.71 | **nous** |
| Étoiles faibles trouvées à 5σ (sur 36715 étoiles Gaia isolées) | 28934 | 28147 | égal |
| Magnitude où la moitié des étoiles est trouvée | 16.89 | 16.74 | **nous** |
| Rapport signal/bruit sur la cible | 46.1 | 48.8 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 5.33 | 5.63 | **nous** |
| FWHM au bord du champ (″) | 5.64 | 5.95 | **nous** |
| Rondeur au centre (1 = rond) | 0.95 | 0.93 | égal |
| Rondeur au bord | 0.87 | 0.87 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 9 | 11 | égal |
| Taches sombres propres à l'image (par degré²) | 3 | 1 | égal |
| Points d'une seule couleur : pixels chauds restants (par degré²) | 2690 | 1414 | Siril |
| Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune) | 0.99 | 1.00 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 23.32 | 23.17 | **nous** |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.061 | 0.073 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.051 | 0.128 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.44 | 0.70 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.90 | 0.97 | |
| Cible : signal / bruit en mag/arcsec² (info) | 20.47 / 24.63 | 20.47 / 24.69 | |
| Durée (min) | 29.2 | 4.2 | |
<!-- /table -->

![m8 express](bench/m8_express.jpg)

### M31 au Seestar S50

<!-- verdict:seestar -->
**À 100 poses, nous allons plus profond ; à 30, Siril garde un petit avantage. Visuellement, Siril montre encore mieux
le grand disque.**
- **Base** : profondeur +0,17 mag, bruit égal à petite échelle et plus faible à grande échelle, étoiles faibles et
  signal/bruit sur le bulbe égaux. **Express** : Siril a 0,10 mag de bruit en moins et va 0,09 mag plus profond.
- **Couleurs des étoiles** plus fidèles chez nous (V−R : 0,069 contre 0,228 en Base).
- **Rondeur au centre** un peu moindre chez nous avec le drizzle 1,5 (0,87 contre 0,91) ; meilleure au bord en Express.
- **Planche** : notre ciel n'est plus violacé, le disque et les bandes de poussière apparaissent, mais plus pâles et
  moins étendus que chez Siril. Le fond est un peu plus plat chez Siril (−0,14 mag).
- **Points d'une seule couleur** : 14 à 25 par degré² chez nous, aucun chez Siril (voir « Ce qui reste »).
<!-- /verdict -->

**Express (30 poses)**

<!-- table:express:seestar -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 30 | 29 | |
| Bruit du fond (mag/arcsec² sur 9.5″, plus haut = moins de bruit) | 22.53 | 22.63 | Siril |
| Bruit du fond à plus grande échelle (38″) | 23.66 | 23.78 | Siril |
| Profondeur : étoile détectable à 5σ (magnitude G) | 15.91 | 16.00 | Siril |
| Étoiles faibles trouvées à 5σ (sur 3945 étoiles Gaia isolées) | 647 | 662 | égal |
| Magnitude où la moitié des étoiles est trouvée | 15.67 | 15.73 | Siril |
| Rapport signal/bruit sur la cible | 20.7 | 20.0 | égal |
| Finesse des étoiles, FWHM au centre (″) | 5.94 | 5.84 | égal |
| FWHM au bord du champ (″) | 5.89 | 5.91 | égal |
| Rondeur au centre (1 = rond) | 0.86 | 0.89 | Siril |
| Rondeur au bord | 0.90 | 0.85 | **nous** |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 0 | 2 | égal |
| Taches sombres propres à l'image (par degré²) | 0 | 0 | égal |
| Points d'une seule couleur : pixels chauds restants (par degré²) | 14 | 0 | Siril |
| Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune) | 0.98 | 0.97 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 21.08 | 21.30 | Siril |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.059 | 0.152 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.191 | 0.262 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.47 | 0.47 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.88 | 0.98 | |
| Cible : signal / bruit en mag/arcsec² (info) | 19.96 / 23.25 | 20.02 / 23.27 | |
| Durée (min) | 1.0 | 0.3 | |
<!-- /table -->

![seestar express](bench/seestar_express.jpg)

**Base (100 poses)**

<!-- table:base:seestar -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 100 | 93 | |
| Bruit du fond (mag/arcsec² sur 9.5″, plus haut = moins de bruit) | 23.16 | 23.19 | égal |
| Bruit du fond à plus grande échelle (38″) | 24.33 | 24.24 | **nous** |
| Profondeur : étoile détectable à 5σ (magnitude G) | 16.72 | 16.55 | **nous** |
| Étoiles faibles trouvées à 5σ (sur 4760 étoiles Gaia isolées) | 1092 | 1065 | égal |
| Magnitude où la moitié des étoiles est trouvée | 16.34 | 16.29 | égal |
| Rapport signal/bruit sur la cible | 35.9 | 37.5 | égal |
| Finesse des étoiles, FWHM au centre (″) | 5.69 | 5.72 | égal |
| FWHM au bord du champ (″) | 5.69 | 5.77 | égal |
| Rondeur au centre (1 = rond) | 0.87 | 0.91 | Siril |
| Rondeur au bord | 0.89 | 0.88 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 2 | 4 | égal |
| Taches sombres propres à l'image (par degré²) | 0 | 0 | égal |
| Points d'une seule couleur : pixels chauds restants (par degré²) | 25 | 0 | Siril |
| Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune) | 1.05 | 1.05 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 21.36 | 21.50 | Siril |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.063 | 0.076 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.069 | 0.228 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.54 | 0.50 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.89 | 0.97 | |
| Cible : signal / bruit en mag/arcsec² (info) | 20.07 / 23.96 | 20.12 / 24.05 | |
| Durée (min) | 1.2 | 0.8 | |
<!-- /table -->

![seestar base](bench/seestar_base.jpg)

### Rosette (Sony A7S défiltré, Newton 130/900, darks de la nuit d'Iris)

<!-- verdict:rosette -->
**Égalité en Base, avec un rendu bien meilleur chez nous ; en Express, nos poses écartées coûtent de la profondeur.**
- **Nuages** : la transparence des poses va de 0,3 à 1,5. Notre moteur garde 16 poses sur 30 en Express et 56 sur 100
  en Base ; Siril garde tout.
- **Base** : profondeur, étoiles faibles et magnitude à 50 % égales (17,71 contre 17,75 ; 7 112 contre 7 121 étoiles),
  avec la moitié des poses. Bruit fin −0,12 mag, signal/bruit sur la nébuleuse −7 %.
- **Express** : avec 16 poses contre 30, Siril va 0,25 mag plus profond et trouve 3 % d'étoiles faibles en plus ;
  signal/bruit sur la nébuleuse −12 %. Les poses nuageuses apportaient donc encore un peu de signal.
- **Couleurs des étoiles** plus fidèles chez nous (B−R 0,034 contre 0,041, V−R 0,052 contre 0,089 en Base).
- **Aucune traînée** le long de la dérive, ni chez nous ni chez Siril (rapport 1,00 à 1,04) : les darks d'Iris ne
  laissent pas de « walking noise », et les traînées des projets du 28.12.24 ont disparu.
- **Planche** : notre anneau H-alpha est rouge franc sur un ciel sombre ; celui de Siril, rose pâle sur un fond gris.
  Les étoiles des coins ont la même coma (le Newton) chez les deux.
<!-- /verdict -->

**Express (30 poses)**

<!-- table:express:rosette -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 16 | 30 | |
| Bruit du fond (mag/arcsec² sur 7.7″, plus haut = moins de bruit) | 23.51 | 23.62 | Siril |
| Bruit du fond à plus grande échelle (31″) | 24.73 | 24.80 | Siril |
| Profondeur : étoile détectable à 5σ (magnitude G) | 17.05 | 17.30 | Siril |
| Étoiles faibles trouvées à 5σ (sur 14788 étoiles Gaia isolées) | 5701 | 5899 | Siril |
| Magnitude où la moitié des étoiles est trouvée | 16.44 | 16.55 | Siril |
| Rapport signal/bruit sur la cible | 4.5 | 5.1 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 5.95 | 5.81 | égal |
| FWHM au bord du champ (″) | 7.50 | 7.36 | égal |
| Rondeur au centre (1 = rond) | 0.76 | 0.76 | égal |
| Rondeur au bord | 0.70 | 0.67 | **nous** |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 10 | 3 | Siril |
| Taches sombres propres à l'image (par degré²) | 4 | 2 | égal |
| Points d'une seule couleur : pixels chauds restants (par degré²) | 77 | 6 | Siril |
| Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune) | 1.00 | 1.00 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 21.97 | 22.00 | égal |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.042 | 0.053 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.078 | 0.103 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.36 | 0.61 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.87 | 0.73 | |
| Cible : signal / bruit en mag/arcsec² (info) | 22.46 / 24.10 | 22.51 / 24.27 | |
| Durée (min) | 3.5 | 1.6 | |
<!-- /table -->

![rosette express](bench/rosette_express.jpg)

**Base (100 poses)**

<!-- table:base:rosette -->
| Mesure | astro_gpu | Siril | Meilleur |
|---|---|---|---|
| Poses empilées | 56 | 100 | |
| Bruit du fond (mag/arcsec² sur 7.7″, plus haut = moins de bruit) | 24.13 | 24.25 | Siril |
| Bruit du fond à plus grande échelle (31″) | 25.13 | 25.25 | Siril |
| Profondeur : étoile détectable à 5σ (magnitude G) | 17.71 | 17.75 | égal |
| Étoiles faibles trouvées à 5σ (sur 15427 étoiles Gaia isolées) | 7112 | 7121 | égal |
| Magnitude où la moitié des étoiles est trouvée | 16.96 | 16.96 | égal |
| Rapport signal/bruit sur la cible | 8.2 | 8.8 | Siril |
| Finesse des étoiles, FWHM au centre (″) | 5.69 | 5.59 | égal |
| FWHM au bord du champ (″) | 7.05 | 7.09 | égal |
| Rondeur au centre (1 = rond) | 0.77 | 0.78 | égal |
| Rondeur au bord | 0.64 | 0.64 | égal |
| Traînées (satellites, avions) restantes | 0 | 0 | égal |
| Taches claires propres à l'image (par degré²) | 8 | 2 | Siril |
| Taches sombres propres à l'image (par degré²) | 5 | 2 | égal |
| Points d'une seule couleur : pixels chauds restants (par degré²) | 76 | 4 | Siril |
| Traînées le long de la dérive (« walking noise ») : bruit le long / bruit moyen (1 = aucune) | 1.04 | 1.03 | égal |
| Fond après dégradé : écart entre les 8×8 zones (mag/arcsec², plus haut = plus plat) | 22.03 | 22.14 | Siril |
| Couleurs des étoiles : erreur B−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.034 | 0.041 | **nous** |
| Couleurs des étoiles : erreur V−R face à Gaia, en BP−RP (plus bas = plus fidèle) | 0.052 | 0.089 | **nous** |
| Écart des couleurs B−R par unité de BP−RP (info : saturation des couleurs) | 1.35 | 0.60 | |
| Couleur Gaia BP−RP rendue blanche (info ; le Soleil : 0,82) | 0.87 | 0.74 | |
| Cible : signal / bruit en mag/arcsec² (info) | 22.49 / 24.77 | 22.58 / 24.94 | |
| Durée (min) | 5.3 | 4.5 | |
<!-- /table -->

![rosette base](bench/rosette_base.jpg)

**Les darks d'Iris vont-elles avec la Rosette ?** (`tools/bench/darkcheck.py` : les pixels chauds et le motif de
chaque master dark, comparés à ceux que les lights montrent d'elles-mêmes, le quart le plus bas de 40 lights en
coordonnées du capteur. Colonne A : sans dark ; colonne C : les 100 fausses darks.)

<!-- darkcheck:rosette -->
| | A : sans dark | B : darks d'Iris | C : fausses darks |
|---|---|---|---|
| Niveau de la master dark (ADU ; noir du capteur 512, fond des lights 528) | 512 (noir) | 508.0 | 520.0 |
| Pixels chauds de la master (excès > 15 ADU ; les lights en montrent 101916) | 0 | 174011 | 106028 |
| Pixels chauds des lights que la master a aussi | – | 99 % | 99 % |
| Pixels chauds de la master absents des lights (creuseraient un trou) | – | 14 % | 1 % |
| Intensité dans les lights / dans la master (pixels chauds forts ; 1 = même température) | – | 0.61 (0.43–0.73) | 0.94 (0.89–0.99) |
| Pixels chauds restants après soustraction (par million de pixels) | 8408 | 44 | 36 |
| Trous creusés par la soustraction (par million) | 0 | 6393 | 412 |
| Motif de lignes / colonnes de la master (ADU ; lights : 0.09 / 0.12) | – | 0.07 / 0.08 | 0.06 / 0.09 |
| Ressemblance de ce motif avec celui des lights (corrélation lignes / colonnes) | – | 0.03 / 0.33 | 0.14 / 0.33 |
| Structure à grande échelle de la master (ADU, p99 − p1 de blocs de 64 px) | – | 4.0 | 8.0 |
<!-- /darkcheck -->

<!-- rosette:darks -->
**Oui.** 99 % des pixels chauds que montrent les lights de la Rosette sont dans la master dark d'Iris : même capteur,
mêmes défauts. Ils y sont 1,6 fois plus forts (rapport 0,61) : le capteur était plus chaud la nuit d'Iris, d'environ
4 °C. Retirés tels quels, ils creuseraient des trous ; mais aucun des deux logiciels ne les soustrait sur ces pixels :
notre moteur les écarte de la pile (2,4 % des pixels), Siril les remplace par leurs voisins (`-cc=dark`). Le niveau
des darks d'Iris, 508 ADU, est le vrai zéro de cet A7S (le fichier annonce 512 : sans dark, chaque light perdrait
4 ADU de ciel). Le motif de lignes et de colonnes du capteur est négligeable (moins de 0,1 ADU). Les fausses darks (C)
avaient, elles, les pixels chauds de la bonne nuit : leur défaut, c'est le ciel qu'elles contiennent.
<!-- /rosette:darks -->

**Ce que font les fausses darks** (essai unique, moteur d'avant les correctifs, Express) : à gauche, la master dark
des 100 photos du ciel prises pour des darks ; à droite, ce qu'elle ajoute à la pile (pile avec ces darks moins pile
sans dark, même grille).

![rosette, fausses darks](bench/rosette_C_damage.jpg)

<!-- rosette:c -->
Les étoiles ont bougé d'une photo à l'autre : la médiane les garde en traits courts et parallèles. Retirés de chaque
light, puis alignés avec la dérive de la monture, ces traits deviennent dans la pile de longues **traînées sombres en
diagonale**, sur tout le champ. Mesuré : le bruit le long de la dérive vaut 1,08 fois le bruit moyen (1,00 à 1,01
avec les darks d'Iris ou sans dark). C'est le défaut que montrait le fond des projets du 28.12.24.
<!-- /rosette:c -->

## Durées

<!-- durations:text -->
Mesurées sur cette machine pendant que d'autres travaux tournaient. Siril calcule sur 8 cœurs (ses durées sont celles
du matin : ses résultats n'ont pas changé). Notre moteur sur la carte cuda:0, **jusqu'à trois calculs du banc en même
temps** pendant cette passe : nos durées sont donc pessimistes. Elles comptent chaque bloc calculé dans la passe
finale ; les catalogues viennent du cache du banc (pour Heart et M8, découpés dans ceux du matin), d'où le peu de
réseau. Sans le réseau, notre calcul prend à peu près le temps de Siril en Base, un peu plus en Express. Non compté :
le premier essai de Heart, arrêté après 25 min d'attente de GAVO (30 tuiles d'étoiles faibles sur 82). M8 a encore
passé 19 min sur ses catalogues (champ très dense, une tuile absente du cache).
<!-- /durations:text -->

<!-- durations -->
| Cas | Niveau | Siril (préparation + fin) | Nous (tout) | dont astrométrie + catalogues (réseau) | Nous sans le réseau |
|---|---|---|---|---|---|
| heart | express | 2.3 min (2.0 + 0.3) | 4.1 min | 1.2 min | 2.9 min |
| heart | base | 5.6 min (5.3 + 0.3) | 5.6 min | 0.4 min | 5.2 min |
| iris | express | 3.2 min (2.8 + 0.4) | 3.7 min | 1.1 min | 2.6 min |
| iris | base | 6.5 min (6.2 + 0.3) | 7.0 min | 0.3 min | 6.6 min |
| m51 | express | 2.9 min (2.4 + 0.6) | 4.6 min | 0.5 min | 4.2 min |
| m51 | base | 9.6 min (9.1 + 0.5) | 8.4 min | 0.1 min | 8.2 min |
| m8 | express | 4.2 min (3.5 + 0.6) | 29.2 min | 18.9 min | 10.3 min |
| seestar | express | 0.3 min (0.2 + 0.1) | 1.0 min | 0.4 min | 0.6 min |
| seestar | base | 0.8 min (0.8 + 0.1) | 1.2 min | 0.1 min | 1.1 min |
| rosette | express | 1.6 min (1.3 + 0.3) | 3.5 min | 0.4 min | 3.1 min |
| rosette | base | 4.5 min (4.2 + 0.3) | 5.3 min | 0.6 min | 4.7 min |
<!-- /durations -->

## Historique (court)

<!-- history -->
1. **Banc du matin** (moteur b43418e, cinq cas) : jeu égal avec Siril sur la profondeur et la finesse, couleurs des
   étoiles 1,3 à 3 fois plus fidèles. Perdu : Iris (15 à 20 % de bruit en plus), le grain fin rouge et bleu, le disque
   de M31 au Seestar, un croissant bleu-cyan sur les étoiles saturées, 57 points chauds d'une couleur par degré² sur M8
   (Siril : 7), et la lenteur du réseau (M8 : 81 min contre 4).
2. **Astrométrie et catalogues hors ligne d'abord** (a15db41) : caches par tuiles du ciel, catalogues locaux de Siril,
   délais courts quand un service se tait.
3. **Huit correctifs du chemin de base** (8d450c1) : rejet calibré sur le ciel de chaque image ; couleurs recalées entre
   elles (dispersion, chromatisme) ; détail plus fin qu'un pixel repris de la luminance ; transparence mesurée dans une
   ouverture fixe ; échantillons du fond pris sur tout le champ ; cœurs saturés de la couleur de leur entourage ;
   pixels chauds reconnus même sans dérive ; textes.
4. **Rosette, sixième cas** : ses « darks » étaient des photos du ciel. Trois calibrations essayées une fois, avec le
   moteur d'avant les correctifs et sur le même ciel : A (sans dark) et B (darks d'Iris) égales à 0,05 mag près sur le
   bruit, la profondeur et les étoiles faibles, chez nous comme chez Siril ; C (les fausses darks) ajoute les traînées
   montrées plus haut. L'utilisateur a choisi B : seul B reste au banc.
5. **Passe finale** (ce document) : les six cas, Express et Base, avec le moteur corrigé ; Siril n'a pas changé.
<!-- /history -->

## Ce qui reste moins bien que Siril

<!-- remaining -->
1. **Des petits points d'une seule couleur** (nouvelle mesure de ce passage) : 14 à 77 par degré² chez nous, et 191
   sur Heart en Base, contre 0 à 6 chez Siril, sur tous les cas (sauf M8, où la mesure se noie dans la nébuleuse). Ce sont des pixels chauds restants, ou des pointes du grain rouge et bleu que notre
   drizzle ne lisse pas (Siril dématrice et interpole) ; la mesure ne les distingue pas. Sur les mêmes poses de la
   Rosette, le moteur d'avant les correctifs en laissait 22 à 23 par degré², le moteur corrigé 77 : un des
   correctifs en laisse passer davantage (à chercher du côté du rejet, `base-1`, dont le taux d'échantillons rejetés
   passe de 0,48 % à 0,04 % sur ces poses, ou du détail fin repris de la luminance, `base-3`). À regarder en premier.
2. **Le bruit fin** (boîtes de 4 pixels) reste 0,03 à 0,13 mag derrière Siril ; à grande échelle, l'écart se réduit ou
   s'inverse (Seestar, Heart en Base).
3. **Iris** : encore 0,1 mag de bruit en plus à toutes les échelles et environ −20 % de signal/bruit sur la nébuleuse
   en Express ; la cause du reste n'est pas trouvée.
4. **Le choix des poses** : quand le ciel est inégal (Rosette, nuages), notre moteur écarte presque la moitié des
   poses ; en Express (30 poses), cela coûte 0,25 mag de profondeur face à Siril qui garde tout. À 100 poses, l'écart
   disparaît.
5. **Rondeur au centre avec le drizzle 1,5** sur les petites étoiles (Seestar, Heart) : 0,87 contre 0,91.
6. **Le grand disque de M31** : visible maintenant, mais plus pâle et moins étendu que chez Siril.
7. **Le réseau** : un champ nouveau télécharge encore son catalogue d'étoiles faibles (G 18,5 à 20,5), que « Classic
   stack » n'emploie pas. CDS muet, GAVO met 20 à 90 s par tuile : 82 tuiles pour Heart, une demi-heure. Le banc a
   découpé ces catalogues dans ceux du matin (`prefetch.py --pool`).
<!-- /remaining -->

## Limites du banc

- **Un seul passage par cas**, sur une machine chargée par d'autres travaux. Un écart de 0,05 mag ou de 3 % est sous
  le seuil de la mesure ; il est noté « égal ». D'un passage à l'autre, la profondeur et le compte d'étoiles faibles
  bougent de ±0,04 mag et ±3 %.
- **Le bruit dépend de l'échelle.** L'interpolation de Siril (dématriçage, puis Lanczos) lisse le bruit d'un pixel à
  l'autre sans ajouter d'information ; notre drizzle garde chaque pixel brut. La grande échelle est la plus juste pour
  comparer l'information.
- **La « cible »** est la région étendue la plus brillante, prise automatiquement sur l'image de Siril.
- **La planéité** est faussée dans les champs pleins de nébulosité (Heart, Iris, M8, Rosette).
- **Les couleurs** sont jugées sur les étoiles, pas sur les nébuleuses ; la référence de blanc diffère d'un logiciel à
  l'autre.
- **Les poses écartées** : notre moteur peut écarter des poses (nuages, flou) ; Siril les garde. Les tableaux le disent
  (« Poses empilées »).

## Refaire le banc

Python : `C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe`, depuis `C:/Users/Nicolas/sandbox/astro`. Tout
ce qui est produit est sous `F:/astro_essais/bench/`.

```
python tools/bench/build.py [CAS ...]                # copies figées, manifestes, état du banc (une fois)
python tools/bench/siril/install_catalogues.py       # catalogues locaux de Siril (une fois)
python tools/bench/calcheck.py [CAS ...]             # contrôle des images de calibration
python tools/bench/siril/run_siril.py CAS NIVEAU     # Siril (8 cœurs)
python tools/bench/ours.py CAS NIVEAU                # notre moteur (cuda:0) ; relancé, seuls les blocs touchés recalculent
python tools/bench/measure.py CAS NIVEAU             # mesures + planche
python tools/bench/darkcheck.py rosette              # les darks d'Iris face aux lights de la Rosette
python tools/bench/report.py fill                    # tableaux de ce document, planches copiées (docs/pilot/bench)
```

La Rosette est un cas « à variantes » (`cases.py`) : sa variante principale B (les darks d'Iris) a les dossiers
habituels (`ours_<niveau>`, `siril_<niveau>`). Les variantes A (sans dark) et C (fausses darks), essayées une fois avec
le moteur d'avant les correctifs, sont rangées dans `rosette/_variants/` (projets, mesures, planche).
