# Banc de qualité : notre moteur face à Siril 1.4

## Banc du 2 novembre : les deux méthodes

Express (30 poses ; M8 : ses 15), les six cas, avec le moteur du jour (drizzle à l'échelle 1 choisie par la règle de
couverture, sélection par poids minimal, taille des étoiles par ajustement de Moffat, rejet « à nous »). Sept images par
cas, mesurées **ensemble sur le même ciel** (le ciel couvert par les sept, les mêmes étoiles Gaia, la même cible, des
boîtes de même taille) :

- **Siril classique** et **Siril drizzle** : leurs résultats existants, Siril n'a pas été relancé ;
- **notre drizzle** : les réglages du jour ;
- **notre drizzle avec le rejet de PixInsight** : le rejet de l'empilement classique appliqué aux gouttes du drizzle
  (Heart et la Rosette : les calculs du matin, repris tels quels) ;
- **notre empilement classique** : avec l'équilibrage des couleurs avant le dématriçage (réglage du jour) ;
- **notre empilement classique sans équilibrage** : le dématriçage exact de Siril ;
- **l'ancienne version** : l'échelle, la sélection et la mesure des étoiles d'hier.

Chaque image passe par la même chaîne (astrométrie, couleur version 4, dégradé de degré 2, sortie) : les couleurs des
étoiles et la planéité sont mesurées sur l'image linéaire finale. Aucun réglage n'a été touché pour le banc. Calculs sur
la carte graphique 0 seule, un à la fois ; aucun échec.

### Ce qu'il faut retenir

1. **Face au drizzle de Siril, notre drizzle gagne** : plus profond de 0,10 à 0,27 mag sur cinq cas (M51 : à égalité),
   bruit fin meilleur de 0,03 à 0,26 mag sur cinq cas (M51 : −0,02), étoiles un peu plus fines sur quatre, couleurs
   égales ou meilleures (Heart : nettement) sauf la Rosette (V−R 0,20 contre 0,11). Il reste en dessous sur un point, le
   signal sur bruit de la cible : de 2 à 16 % plus bas sur cinq cas, +1 % sur la Rosette.
2. **Face à Siril classique, notre drizzle est derrière** : 0,06 à 0,11 mag de profondeur sur quatre cas (la Rosette :
   0,23), 0,02 à 0,12 mag de bruit fin sur les six, un signal sur bruit plus bas de 3 à 20 % (sauf le Seestar, +7 %), et
   des points d'une seule couleur (10 à 24 par degré² contre 0 à 6 chez Siril classique hors M8). Il est devant sur M8
   (+0,11 mag) et a de loin les meilleures couleurs d'étoiles sur Heart (B−R 0,09 contre 0,22), le Seestar et M8.
3. **Notre classique sans équilibrage rejoint Siril classique** : profondeur à 0,11 mag près, bruit fin à 0,01 mag près
   (la Rosette : +0,08 mag), puisque c'est le même dématriçage. Sur Heart il est orangé (erreur B−R 0,238 contre 0,093
   pour notre drizzle ; planéité 0,9 mag sous celle de Siril).
4. **L'équilibrage avant le dématriçage** corrige ce défaut (Heart : B−R 0,153, planéité +0,7 mag), améliore le signal
   sur bruit de la cible sur les six cas (+1 à +9 %) et l'erreur V−R sur quatre (−0,04 à −0,07), mais **il coûte de la
   profondeur** : −0,57 mag sur Heart, −0,03 à −0,08 mag sur quatre autres cas (M51 : +0,08 mag, et meilleur partout),
   et des étoiles 1 à 3 % plus larges. Il ne règle pas M8, où les couleurs d'étoiles classiques restent mauvaises (V−R
   0,23 contre 0,05 pour notre drizzle).
5. **Le rejet de PixInsight appliqué à notre drizzle** ne change presque rien sur Heart, Iris et M51, et **ne retire pas
   les points d'une seule couleur** (24 contre 23 sur Heart, 17 contre 17 sur M51, 2 680 contre 2 638 sur M8). Il
   répare la Rosette (V−R de 0,20 à 0,07, planéité +0,7 mag), mais il est un peu moins bon sur M8 (couleurs : B−R 0,063
   contre 0,056) et sur le Seestar (B−R 0,136 contre 0,105 ; signal sur bruit −3 %). Il coûte 1,6 à 4 fois plus de
   temps (M8 : 17 minutes contre 4).
6. **Laquelle de nos deux méthodes est la meilleure ?** (nombre de mesures du critère où l'une bat l'autre, au-delà de la
   variation d'un passage à l'autre) Le drizzle devant sur **Heart** (7 contre 2 : profondeur, bruit large, étoiles
   faibles, couleurs, planéité ; mais le classique sans équilibrage est le plus profond) et **M8** (5 contre 1). Le
   classique devant sur **Iris** (6 contre 2), **M51** (5 contre 0), le **Seestar** (7 contre 1) et la **Rosette**
   (6 contre 0) : profondeur, bruit fin, signal sur bruit, et points d'une seule couleur (2 à 14 contre 10 à 24 par
   degré² hors M8).
7. **Le conseil de méthode du moteur** donne : Heart classique, Iris drizzle, M51 classique, M8 drizzle, Seestar drizzle,
   Rosette classique. **Il tombe sur la meilleure de nos deux méthodes dans 3 cas sur 6** (M51, M8, Rosette). Il se
   trompe sur Heart (conseille le classique, le drizzle fait mieux, sauf pour le classique sans équilibrage), sur Iris et
   sur le Seestar (conseille le drizzle, le classique fait mieux). La variation de couverture que lit la règle
   (17,9 % au Seestar, 20,0 % à Iris, 21,1 à 21,8 % à Heart, M51 et la Rosette ; plus grande valeur mesurée, seuil de
   20 %) tombe à quelques points du seuil sur chaque cas : le seuil seul ne sépare pas les cas.
8. **Par rapport à l'ancienne version**, les réglages du jour ne font pas moins bien : un peu mieux partout (profondeur
   +0,01 à +0,07 mag, bruit fin +0,02 à +0,12 mag, signal sur bruit +1 à +13 %, la Rosette en tête : l'ancienne sélection
   n'y gardait que 16 poses sur 30, la nouvelle 26), mais des étoiles 1 à 4 % plus larges (Heart : 12,55″ contre 12,07″).
9. **Durées** (calibration, alignement, empilement ; machine partagée) : notre drizzle 0,9 à 4,2 minutes, notre
   classique 1,0 à 16 minutes (M8 : 16 minutes pour 15 poses de 26 Mpx), Siril classique 0,2 à 3,5 minutes.
10. **Taux de rejet** : Siril et nos empilements classiques rejettent 0,8 à 1,6 % des échantillons (repère de Siril :
    0,1 à 0,5 %) ; le rejet « à nous » de notre drizzle 0,001 à 0,045 % : ce n'est pas la même mesure (il ne retire que
    les valeurs aberrantes qu'il prévoit), ne pas les comparer.
11. **Sur les planches** (100 %, même flux, même étirement) : notre drizzle et l'ancienne version montrent sur les zones
    faibles de la Rosette une texture magenta que le rejet de PixInsight atténue ; l'empilement classique sans
    équilibrage est orange sur Heart et jaune-vert autour de la galaxie du Seestar (bleu avec l'équilibrage) ; le drizzle de
    Siril a un fond très inégal sur M51 (planéité 19,9 contre 23).

Lecture des tableaux : un tableau par cas ; **en gras** le meilleur des cinq empilements de chez nous sur la ligne (les
deux Siril sont donnés pour repère). Le ciel commun aux sept images est un peu plus petit que celui des tableaux
précédents : quelques valeurs de Siril classique diffèrent de la deuxième décimale. « Point blanc » : l'indice Gaia
(BP−RP) d'une étoile qui sort sans écart bleu−rouge (environ 0,8 pour une étoile de type solaire ; un écart dit un fond
de couleur trop chaud ou trop froid). Siril a écarté une pose du Seestar (29 gardées). Les mesures brutes sont dans
`F:/astro_essais/bench/<cas>/today_express.json`, les tableaux et les verdicts dans
`docs/pilot/bench/two_methods_tables.md` et `two_methods_verdicts.json`, l'outil dans `tools/bench/two_methods.py`.

### Les planches (100 %, même flux, même étirement)

`docs/pilot/shots/2026-11-02/bench/` : `heart.png`, `iris.png`, `m51.png`, `m8.png`, `seestar.png`, `rosette.png`. Sept
colonnes (dans l'ordre des tableaux), deux recadrages de 400 × 400 pixels de Siril à 100 % : le champ d'étoiles le plus
dense et une zone faible. Ce sont les images linéaires finales, mises à la même échelle de flux (le flux des mêmes étoiles
Gaia) et montrées avec le même étirement (arcsinh) ; le fond de chaque couleur est ramené au même niveau, pour que la
planche montre le grain, les points colorés et les étoiles, pas un voile de couleur (les voiles sont dans la ligne
« planéité » des tableaux).

### Les tableaux

#### Heart (30 poses)

| | Siril classique | Siril drizzle | Notre drizzle | Notre drizzle, rejet PixInsight | Notre classique | Notre classique sans équilibrage | Ancienne version |
|---|---:|---:|---:|---:|---:|---:|---:|
| Profondeur (mag, 5σ ; plus haut = plus profond) | 16,36 | 16,03 | 16,27 | 16,27 | 15,81 | **16,38** | 16,22 |
| Bruit fin (mag/arcsec² ; plus haut = moins de bruit) | 24,61 | 24,25 | 24,51 | 24,50 | 24,53 | **24,61** | 24,48 |
| Bruit large (mag/arcsec²) | 25,30 | 25,01 | 25,29 | 25,28 | 25,24 | **25,32** | 25,28 |
| Signal sur bruit de la cible | 4,8 | 4,9 | 4,5 | 4,5 | **5,0** | 4,6 | 4,4 |
| Étoiles faibles trouvées | 21784 | 21322 | 21640 | 21687 | 20504 | **21847** | 21596 |
| FWHM au centre (″ ; plus bas = plus fines) | 13,10 | 12,34 | 12,55 | 12,57 | 13,36 | 13,01 | **12,07** |
| Rondeur (1 = rondes) | 0,91 | 0,90 | 0,91 | 0,92 | **0,93** | 0,92 | 0,91 |
| Points d'une seule couleur (par degré²) | 1 | 75 | 24 | 23 | 2 | **1** | 21 |
| Erreur de couleur B−R (plus bas = plus fidèle) | 0,217 | 0,263 | **0,093** | 0,096 | 0,153 | 0,238 | 0,093 |
| Erreur de couleur V−R | 0,326 | 0,579 | 0,184 | **0,173** | 0,283 | 0,329 | 0,188 |
| Point blanc B−R (BP−RP où B−R = 0 ; Gaia : environ 0,8) | 0,931 | 0,949 | 0,933 | 0,940 | 0,964 | 0,949 | 0,923 |
| Planéité du fond (mag/arcsec² ; plus haut = plus plat) | 24,59 | 24,25 | 24,67 | **24,70** | 24,37 | 23,69 | 24,59 |
| Poses gardées | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Taux de rejet (% des échantillons) | 0,941 | 0,990 | 0,014 | 1,060 | 1,060 | 1,097 | 0,014 |
| Durée de la préparation (min) | 2,0 | 1,3 | 2,5 | 4,1 | 3,5 | 3,5 | 2,3 |

#### Iris (30 poses)

| | Siril classique | Siril drizzle | Notre drizzle | Notre drizzle, rejet PixInsight | Notre classique | Notre classique sans équilibrage | Ancienne version |
|---|---:|---:|---:|---:|---:|---:|---:|
| Profondeur (mag, 5σ ; plus haut = plus profond) | 17,38 | 17,12 | 17,31 | 17,32 | 17,37 | **17,44** | 17,30 |
| Bruit fin (mag/arcsec² ; plus haut = moins de bruit) | 23,85 | 23,69 | 23,72 | 23,72 | **23,86** | 23,86 | 23,68 |
| Bruit large (mag/arcsec²) | 25,22 | 25,07 | 25,12 | 25,12 | 25,17 | **25,21** | 25,10 |
| Signal sur bruit de la cible | 3,0 | 2,8 | 2,4 | 2,4 | **2,7** | 2,6 | 2,4 |
| Étoiles faibles trouvées | 4633 | 4493 | 4640 | 4628 | 4681 | **4735** | 4631 |
| FWHM au centre (″ ; plus bas = plus fines) | 6,12 | 6,55 | 6,24 | 6,26 | 6,28 | **6,08** | 6,15 |
| Rondeur (1 = rondes) | 0,89 | 0,89 | 0,88 | 0,88 | **0,90** | 0,89 | 0,88 |
| Points d'une seule couleur (par degré²) | 2 | 17 | 21 | 15 | 14 | **7** | 26 |
| Erreur de couleur B−R (plus bas = plus fidèle) | 0,058 | 0,042 | 0,046 | 0,046 | 0,055 | 0,089 | **0,045** |
| Erreur de couleur V−R | 0,069 | 0,067 | **0,058** | 0,062 | 0,067 | 0,072 | 0,058 |
| Point blanc B−R (BP−RP où B−R = 0 ; Gaia : environ 0,8) | 0,869 | 0,906 | 0,914 | 0,921 | 0,923 | 0,917 | 0,915 |
| Planéité du fond (mag/arcsec² ; plus haut = plus plat) | 23,26 | 23,15 | 23,43 | 23,48 | 23,53 | **23,55** | 23,37 |
| Poses gardées | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Taux de rejet (% des échantillons) | 0,892 | 0,788 | 0,045 | 1,403 | 1,403 | 1,347 | 0,044 |
| Durée de la préparation (min) | 2,8 | 1,6 | 2,5 | 4,1 | 3,6 | 3,5 | 2,3 |

#### M51 (30 poses)

| | Siril classique | Siril drizzle | Notre drizzle | Notre drizzle, rejet PixInsight | Notre classique | Notre classique sans équilibrage | Ancienne version |
|---|---:|---:|---:|---:|---:|---:|---:|
| Profondeur (mag, 5σ ; plus haut = plus profond) | 18,27 | 18,20 | 18,21 | 18,20 | **18,28** | 18,20 | 18,20 |
| Bruit fin (mag/arcsec² ; plus haut = moins de bruit) | 24,65 | 24,59 | 24,57 | 24,56 | **24,70** | 24,64 | 24,54 |
| Bruit large (mag/arcsec²) | 25,91 | 25,87 | 25,90 | 25,87 | **25,91** | 25,86 | 25,88 |
| Signal sur bruit de la cible | 4,7 | 4,4 | 4,2 | 4,1 | **4,8** | 4,5 | 4,1 |
| Étoiles faibles trouvées | 1271 | 1237 | 1237 | 1239 | **1262** | 1225 | 1245 |
| FWHM au centre (″ ; plus bas = plus fines) | 6,88 | 7,09 | 6,99 | 6,99 | 6,94 | **6,86** | 6,95 |
| Rondeur (1 = rondes) | 0,82 | 0,83 | **0,84** | 0,83 | 0,83 | 0,83 | 0,83 |
| Points d'une seule couleur (par degré²) | 4 | 16 | 17 | 17 | 6 | **5** | 17 |
| Erreur de couleur B−R (plus bas = plus fidèle) | 0,021 | 0,024 | 0,025 | 0,027 | 0,025 | **0,024** | 0,025 |
| Erreur de couleur V−R | 0,047 | 0,041 | 0,033 | **0,033** | 0,034 | 0,038 | 0,035 |
| Point blanc B−R (BP−RP où B−R = 0 ; Gaia : environ 0,8) | 0,424 | 0,458 | 0,870 | 0,871 | 0,881 | 0,878 | 0,871 |
| Planéité du fond (mag/arcsec² ; plus haut = plus plat) | 22,99 | 19,92 | 23,16 | 23,19 | **23,28** | 23,24 | 23,21 |
| Poses gardées | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Taux de rejet (% des échantillons) | 0,804 | 0,596 | 0,025 | 0,814 | 0,814 | 0,852 | 0,024 |
| Durée de la préparation (min) | 2,4 | 2,1 | 3,9 | 6,4 | 5,5 | 5,4 | 3,6 |

#### M8 (15 poses)

| | Siril classique | Siril drizzle | Notre drizzle | Notre drizzle, rejet PixInsight | Notre classique | Notre classique sans équilibrage | Ancienne version |
|---|---:|---:|---:|---:|---:|---:|---:|
| Profondeur (mag, 5σ ; plus haut = plus profond) | 17,66 | 17,50 | **17,78** | 17,75 | 17,58 | 17,62 | 17,74 |
| Bruit fin (mag/arcsec² ; plus haut = moins de bruit) | 23,97 | 23,86 | 23,95 | 23,94 | 23,95 | **23,97** | 23,93 |
| Bruit large (mag/arcsec²) | 24,10 | 24,00 | 24,08 | 24,07 | 24,07 | **24,10** | 24,07 |
| Signal sur bruit de la cible | 46,0 | 45,0 | 44,2 | 44,5 | **48,1** | 47,5 | 43,5 |
| Étoiles faibles trouvées | 25183 | 25082 | 25601 | **25711** | 24884 | 24901 | 25646 |
| FWHM au centre (″ ; plus bas = plus fines) | 5,61 | 5,68 | 5,45 | 5,46 | 5,65 | 5,58 | **5,33** |
| Rondeur (1 = rondes) | 0,93 | 0,92 | **0,95** | 0,95 | 0,93 | 0,93 | 0,95 |
| Points d'une seule couleur (par degré²) | 1704 | 7159 | 2680 | 2638 | 1655 | **1425** | 2576 |
| Erreur de couleur B−R (plus bas = plus fidèle) | 0,075 | 0,057 | 0,056 | 0,063 | 0,086 | 0,087 | **0,053** |
| Erreur de couleur V−R | 0,133 | 0,076 | **0,049** | 0,057 | 0,230 | 0,296 | 0,051 |
| Point blanc B−R (BP−RP où B−R = 0 ; Gaia : environ 0,8) | 0,972 | 0,972 | 0,905 | 0,904 | 0,964 | 0,961 | 0,904 |
| Planéité du fond (mag/arcsec² ; plus haut = plus plat) | 23,17 | 23,09 | 23,25 | **23,30** | 22,96 | 22,90 | 23,19 |
| Poses gardées | 15 | 15 | 15 | 15 | 15 | 15 | 15 |
| Taux de rejet (% des échantillons) | 1,222 | 1,177 | 0,014 | 1,167 | 1,167 | 1,215 | 0,014 |
| Durée de la préparation (min) | 3,5 | 4,1 | 4,2 | 17,0 | 16,3 | 16,5 | 6,8 |

#### Seestar (30 poses)

| | Siril classique | Siril drizzle | Notre drizzle | Notre drizzle, rejet PixInsight | Notre classique | Notre classique sans équilibrage | Ancienne version |
|---|---:|---:|---:|---:|---:|---:|---:|
| Profondeur (mag, 5σ ; plus haut = plus profond) | 16,06 | 15,85 | 15,95 | 15,94 | 15,99 | **16,03** | 15,93 |
| Bruit fin (mag/arcsec² ; plus haut = moins de bruit) | 22,63 | 22,47 | 22,55 | 22,53 | 22,63 | **22,64** | 22,52 |
| Bruit large (mag/arcsec²) | 23,80 | 23,62 | 23,69 | 23,66 | 23,69 | **23,74** | 23,65 |
| Signal sur bruit de la cible | 19,1 | 21,0 | 20,5 | 19,8 | **21,6** | 20,5 | 19,2 |
| Étoiles faibles trouvées | 647 | 607 | 612 | 616 | 627 | **634** | 621 |
| FWHM au centre (″ ; plus bas = plus fines) | 5,82 | 6,47 | 6,10 | 6,12 | 5,94 | **5,83** | 5,94 |
| Rondeur (1 = rondes) | 0,89 | 0,87 | 0,87 | 0,86 | 0,91 | **0,91** | 0,86 |
| Points d'une seule couleur (par degré²) | 0 | 7 | 10 | 10 | **2** | 5 | 5 |
| Erreur de couleur B−R (plus bas = plus fidèle) | 0,158 | 0,147 | 0,105 | 0,136 | 0,120 | **0,090** | 0,133 |
| Erreur de couleur V−R | 0,259 | 0,239 | 0,188 | 0,159 | **0,116** | 0,175 | 0,166 |
| Point blanc B−R (BP−RP où B−R = 0 ; Gaia : environ 0,8) | 0,978 | 0,949 | 0,966 | 0,946 | 1,073 | 0,995 | 0,958 |
| Planéité du fond (mag/arcsec² ; plus haut = plus plat) | 21,30 | 21,37 | 21,06 | 21,07 | **21,39** | 21,35 | 21,01 |
| Poses gardées | 29 | 30 | 30 | 30 | 30 | 30 | 30 |
| Taux de rejet (% des échantillons) | 0,919 | 0,630 | 0,001 | 0,920 | 0,920 | 0,959 | 0,001 |
| Durée de la préparation (min) | 0,2 | 0,3 | 0,9 | 1,1 | 1,0 | 1,0 | 0,8 |

#### Rosette (30 poses)

| | Siril classique | Siril drizzle | Notre drizzle | Notre drizzle, rejet PixInsight | Notre classique | Notre classique sans équilibrage | Ancienne version |
|---|---:|---:|---:|---:|---:|---:|---:|
| Profondeur (mag, 5σ ; plus haut = plus profond) | 17,36 | 16,96 | 17,12 | 17,13 | 17,17 | **17,25** | 17,05 |
| Bruit fin (mag/arcsec² ; plus haut = moins de bruit) | 23,64 | 23,43 | 23,60 | 23,59 | 23,71 | **23,72** | 23,47 |
| Bruit large (mag/arcsec²) | 24,82 | 24,65 | 24,82 | 24,82 | 24,82 | **24,89** | 24,75 |
| Signal sur bruit de la cible | 4,9 | 4,7 | 4,7 | 4,8 | **5,4** | 5,0 | 4,2 |
| Étoiles faibles trouvées | 5802 | 5458 | 5608 | 5601 | 5659 | **5783** | 5514 |
| FWHM au centre (″ ; plus bas = plus fines) | 5,81 | 5,99 | 6,02 | 6,02 | 6,02 | **5,87** | 5,96 |
| Rondeur (1 = rondes) | 0,76 | 0,77 | 0,76 | 0,76 | **0,77** | 0,77 | 0,76 |
| Points d'une seule couleur (par degré²) | 6 | 17 | 17 | 17 | 7 | **6** | 22 |
| Erreur de couleur B−R (plus bas = plus fidèle) | 0,053 | 0,054 | 0,056 | **0,041** | 0,054 | 0,056 | 0,060 |
| Erreur de couleur V−R | 0,102 | 0,109 | 0,201 | **0,072** | 0,112 | 0,152 | 0,200 |
| Point blanc B−R (BP−RP où B−R = 0 ; Gaia : environ 0,8) | 0,734 | 0,759 | 0,859 | 0,867 | 0,894 | 0,887 | 0,853 |
| Planéité du fond (mag/arcsec² ; plus haut = plus plat) | 21,95 | 21,79 | 21,29 | 21,95 | 22,07 | **22,10** | 21,25 |
| Poses gardées | 30 | 30 | 26 | 26 | 26 | 26 | 16 |
| Taux de rejet (% des échantillons) | 0,910 | 0,841 | 0,044 | 1,603 | 1,603 | 1,622 | 0,039 |
| Durée de la préparation (min) | 1,3 | 1,6 | 2,7 | 4,3 | 3,6 | 3,7 | 2,5 |

> **2 novembre, nuit : banc croisé de l'empilement v2.** Les six changements de la spec (docs/spec/STACKING.md) ont
> été essayés un par un, tous ensemble et tous sauf un, sur les six cas, contre l'ancienne version et contre Siril.
> Ce document remplace les tableaux de la passe du 1er novembre (moteur avec le correctif n° 3, retiré depuis) ; les
> projets et la recette Siril n'ont pas changé.

## En bref

<!-- enbref -->
Le banc croisé de l'empilement v2 : 19 configurations sur les six cas en Express (30 poses), mesurées ensemble sur le
même ciel, puis l'ancienne version (OLD), la configuration retenue et V2 en Base (100 poses).

- **Aucun changement ne fait gagner nettement le drizzle.** Le plus utile est **S1** (taille des étoiles par
  ajustement de Moffat, échelle et goutte choisies par la règle de couverture) : bruit fin plus faible de 0,02 à 0,04
  mag, signal/bruit +2 à +6 %, pour des étoiles 1 à 4 % plus larges. **C'est la configuration retenue** par le critère
  (3 cases moins bien qu'OLD sur 65, toutes petites). Les autres changements sont neutres (S2, S3, S5, S6) ou nuisent
  en Express (la sélection k × σ, S4, sans la FWHM de Moffat).
- **En Base**, S1 est identique à OLD sur Heart et le Seestar (la règle y garde l'échelle 1,5), un peu meilleur sur
  Iris et M51 (+0,02 à +0,03 mag), moins bon sur la Rosette (échelle 1 : −0,08 mag, couleurs des étoiles et fond moins
  bons). Le critère n'y est donc tenu que sur 4 cas sur 5.
- **La méthode Classique rejoint Siril** (son bruit à 0,02 mag près ; +0,11 à +0,17 mag de profondeur sur Heart, Iris,
  le Seestar et la Rosette, points d'une seule couleur au niveau de Siril) **mais abîme les couleurs** (Heart orangé,
  erreur de couleur 2,5 fois plus forte) : elle ne passe pas le critère tant que ce n'est pas corrigé.
- **Face à Siril, avec S1, en Base** : devant sur Heart (+0,08 mag) et le Seestar (+0,14), derrière sur Iris (−0,13),
  la Rosette (−0,12) et M51 (−0,04) ; le bruit fin reste 0,05 à 0,15 mag derrière partout ; les couleurs des étoiles
  sont plus fidèles sauf sur la Rosette ; les points d'une seule couleur restent (9 à 125 par degré², Siril 0 à 5).
- **Deux défauts du moteur trouvés** (non corrigés, signalés) : un message avec « σ » tue l'étape sous Windows
  (encodage du journal), et la méthode Classique déclenche parfois une erreur CUDA qui fait échouer tout le reste du
  calcul. Voir « Ce qui reste ».
<!-- /enbref -->

## Empilement v2 : le banc croisé

### Ce qu'on a comparé

Les six changements de la spec (docs/spec/STACKING.md) sont dans le moteur, chacun derrière un interrupteur, à côté
de l'ancienne version. Le banc les essaie **un par un, tous ensemble, et tous sauf un**, sur les six cas en Express
(30 poses ; M8 : ses 15), puis la meilleure configuration en Base (100 poses).

| Code | Ce qui change par rapport à l'ancienne version |
|---|---|
| OLD | rien : l'ancienne version, tous les interrupteurs à leur valeur par défaut |
| OLD-bis | rien, sauf un recalage à peine différent (seuil de détection 7,9 σ au lieu de 8) : mesure le bruit d'un passage à l'autre |
| S1 | taille des étoiles par ajustement de Moffat, et échelle et goutte du drizzle choisies par la règle de couverture (DrizzlePac) |
| S2 | rejet de Hubble (AstroDrizzle `driz_cr`) |
| S3 | pondération des poses par l'inverse de la variance, couleur par couleur |
| S4 | sélection des poses en k × σ (FWHM, rondeur, transparence ; poids minimal 5 %), comme Siril et WBPP |
| S4b | la même sélection sans le filtre de transparence |
| S5 | normalisation locale, couleur par couleur, sur une référence faite des 20 meilleures poses |
| S6 | recalage du rouge et du bleu sur le vert par blocs de poses, dans le drizzle |
| S7 | méthode « Classique » : dématriçage RCD, recalage Lanczos-4, rejet choisi selon le nombre de poses (WBPP) |
| V2 | S1 à S6 ensemble, drizzle |
| V2-auto | S1 à S6 ensemble, avec la méthode que conseille la règle de couverture (Classique pour Heart seulement ; drizzle ailleurs) |
| V2-C | S1 à S6 ensemble, en méthode Classique partout (ajoutée parce que S7 seul était de loin le plus fort) |
| V2-Sk | V2 avec le changement k remis à l'ancienne version (V2-S1 … V2-S6) |

### Comment on a mesuré

- **Mêmes fichiers, même ciel.** Chaque cas a un projet `v2x_express` (et `v2x_base`) sous
  `F:/astro_essais/bench/<cas>/`, avec une chaîne « Classic stack » par configuration ; la calibration est calculée
  une fois, chaque recalage une fois.
- **Toutes les images d'un cas sont mesurées ensemble** : le ciel couvert par toutes, la même liste d'étoiles Gaia, la
  même cible, des boîtes de même taille sur le ciel. Les écarts entre configurations sont donc ceux des images, pas
  ceux de la mesure.
- **Siril n'a pas été relancé** : ce sont ses résultats du matin (même recette, mêmes fichiers).
- **OLD n'est plus tout à fait le moteur des anciens tableaux** : le correctif n° 3 (détail fin) a été retiré à 18 h 45.
  OLD est donc le moteur actuel, interrupteurs coupés.
- Le catalogue des étoiles très faibles (G 18,5 à 20,5) n'est téléchargé que pour OLD : la chaîne « Classic stack » ne
  s'en sert pas, et les mesures prennent celui d'OLD pour toutes les images.
- **Les durées** sont mesurées sur une machine partagée (le client, Photoshop, plusieurs calculs du banc en même
  temps) : elles donnent un ordre de grandeur.

### Le critère de choix

Le principe, fixé avant la passe :

> **Ne faire moins bien qu'OLD nulle part au-delà de la variation d'un passage à l'autre, et faire mieux que Siril là où
> c'est possible.**

Sa mise en œuvre : 11 mesures × 6 cas = 65 cases par configuration (sans les points d'une seule couleur de M8). Une case
est « moins bien » qu'OLD quand la configuration perd plus de 0,04 mag de profondeur ou de bruit (aux deux échelles),
3 % de signal/bruit sur la cible ou d'étoiles faibles trouvées, 3 % de finesse (FWHM), 0,02 de rondeur, 0,005 d'erreur
de couleur B−R ou V−R, 0,1 mag de planéité, ou 15 % (et au moins 3 par degré²) de points d'une seule couleur ; ces
seuils sont relevés au besoin à la variation mesurée par OLD-bis (seule la couleur V−R l'a été, à 0,009). Même règle,
dans l'autre sens, pour « mieux », et face à Siril.

On retient la configuration qui a **le moins de cases moins bien qu'OLD**, puis, à égalité, **le plus de cases mieux
que Siril**, parmi celles qui font mieux qu'OLD au moins quelque part. Cette dernière condition a été ajoutée en
voyant le résultat : la règle brute plaçait en tête S6 et S5 seuls, qui ne changent rien de mesurable (aucune case
mieux qu'OLD). Le taux de rejet, le nombre de poses et la durée sont donnés pour information.

### Ce que donne chaque changement (Express)

Écart à OLD, sur les six cas : la médiane, puis entre crochets le pire et le meilleur cas ; ▲ = nombre de cas
meilleurs qu'OLD au-delà de la variation d'un passage à l'autre, ▼ = nombre de cas moins bons. Magnitudes et
pourcentages positifs = mieux, sauf la FWHM, les points d'une seule couleur et l'erreur de couleur (positif = moins
bien). Les points d'une seule couleur de M8 ne comptent pas (nébulosité partout).

**Chaque changement seul** (et les ensembles), face à OLD :

<!-- crossed-alone:express -->
| | Profondeur | Bruit fin | Bruit large | S/B cible | Étoiles faibles | FWHM | Rondeur | Points 1 couleur | Couleur B−R | Couleur V−R | Planéité |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OLD-bis | −0,00 [−0,03 ; +0,03] | −0,00 [−0,00 ; +0,00] | +0,00 [−0,03 ; +0,00] | +0 % [−0 ; +0] | −0 % [−0 ; +1] | +0 % [−0 ; +0] | +0,00 [−0,00 ; +0,00] | +0 [−0 ; +1] | +0,000 [−0,001 ; +0,001] | −0,001 [−0,009 ; +0,000] | −0,00 [−0,01 ; +0,00] |
| S1 | +0,01 [+0,00 ; +0,03] | +0,04 [+0,02 ; +0,04] 2▲ | +0,01 [+0,00 ; +0,02] | +3 % [+2 ; +6] 4▲ | +0 % [−1 ; +1] | +2 % [+1 ; +4] 1▼ | +0,00 [+0,00 ; +0,01] | +0 [−4 ; +3] 1▲ | +0,001 [−0,001 ; +0,003] | −0,002 [−0,003 ; +0,030] 2▼ | +0,06 [−0,05 ; +0,08] |
| S2 | −0,02 [−0,07 ; +0,01] 1▼ | −0,01 [−0,03 ; −0,00] | −0,01 [−0,03 ; −0,00] | −0 % [−1 ; +1] | −0 % [−1 ; +1] | +0 % [−0 ; +1] | +0,00 [+0,00 ; +0,00] | +3 [−0 ; +15] 2▼ | +0,003 [−0,001 ; +0,012] 1▼ | −0,000 [−0,011 ; +0,071] 2▲ 1▼ | −0,01 [−0,08 ; +0,08] |
| S3 | −0,01 [−0,04 ; +0,02] 1▼ | +0,01 [−0,00 ; +0,01] | +0,00 [−0,05 ; +0,01] 1▼ | +1 % [−0 ; +4] 1▲ | −0 % [−2 ; +1] | +1 % [+0 ; +1] | −0,00 [−0,01 ; +0,00] | +1 [−2 ; +3] | +0,001 [−0,007 ; +0,012] 1▲ 1▼ | −0,001 [−0,040 ; +0,081] 1▲ 1▼ | −0,00 [−0,02 ; +0,12] 1▲ |
| S4 | −0,05 [−0,22 ; +0,04] 1▲ 3▼ | −0,05 [−0,15 ; +0,05] 1▲ 3▼ | −0,01 [−0,18 ; +0,04] 2▼ | −5 % [−14 ; +8] 1▲ 4▼ | −0 % [−5 ; +2] 2▼ | −1 % [−2 ; +0] | +0,00 [−0,01 ; +0,02] | −1 [−8 ; +1] 1▲ | +0,001 [−0,020 ; +0,045] 1▲ 2▼ | −0,004 [−0,134 ; −0,000] 2▲ | +0,00 [−0,03 ; +0,64] 1▲ |
| S4b | −0,05 [−0,22 ; +0,04] 1▲ 3▼ | −0,05 [−0,15 ; +0,05] 1▲ 3▼ | −0,01 [−0,18 ; +0,04] 2▼ | −5 % [−14 ; +8] 1▲ 4▼ | −0 % [−5 ; +2] 2▼ | −1 % [−2 ; +0] | +0,00 [−0,01 ; +0,02] | −1 [−8 ; +1] 1▲ | +0,001 [−0,020 ; +0,045] 1▲ 2▼ | −0,004 [−0,134 ; −0,000] 2▲ | +0,00 [−0,03 ; +0,64] 1▲ |
| S5 | −0,01 [−0,03 ; +0,00] | −0,00 [−0,00 ; +0,00] | +0,00 [−0,02 ; +0,01] | +0 % [−11 ; +2] 1▼ | +0 % [−1 ; +1] | +0 % [−0 ; +0] | +0,00 [−0,00 ; +0,00] | +0 [−1 ; +3] | −0,000 [−0,001 ; +0,022] 1▼ | +0,000 [−0,004 ; +0,013] 1▼ | −0,00 [−0,03 ; +0,02] |
| S6 | +0,00 [−0,04 ; +0,03] | +0,00 [−0,01 ; +0,01] | −0,00 [−0,03 ; +0,01] | −0 % [−1 ; +1] | +0 % [−1 ; +1] | +0 % [−0 ; +0] | +0,00 [−0,00 ; +0,00] | −0 [−1 ; +2] | +0,000 [−0,000 ; +0,026] 1▼ | +0,000 [−0,001 ; +0,003] | −0,00 [−0,02 ; +0,02] |
| S7 | +0,12 [−0,13 ; +0,17] 4▲ 1▼ | +0,13 [+0,04 ; +0,18] 6▲ | +0,07 [−0,02 ; +0,12] 4▲ | +9 % [+8 ; +11] 6▲ | +2 % [−2 ; +4] 2▲ | −1 % [−2 ; +8] 2▼ | +0,01 [−0,02 ; +0,05] 1▲ | −15 [−20 ; −0] 4▲ | +0,016 [−0,009 ; +0,143] 1▲ 3▼ | +0,009 [−0,034 ; +0,227] 1▲ 3▼ | +0,10 [−0,97 ; +0,90] 3▲ 2▼ |
| V2 | +0,00 [−0,13 ; +0,02] 2▼ | +0,02 [+0,00 ; +0,07] 1▲ | +0,00 [−0,02 ; +0,03] | +1 % [−8 ; +9] 2▲ 1▼ | −0 % [−3 ; +1] 1▼ | +3 % [+1 ; +5] 1▼ | +0,01 [−0,01 ; +0,01] | −0 [−4 ; +5] 1▲ 1▼ | +0,006 [−0,001 ; +0,012] 3▼ | −0,001 [−0,041 ; +0,007] 1▲ | +0,04 [−0,08 ; +0,08] |
| V2-auto | +0,01 [−0,13 ; +0,16] 1▲ 2▼ | +0,02 [+0,00 ; +0,13] 2▲ | +0,01 [−0,02 ; +0,06] 1▲ | +1 % [−8 ; +10] 2▲ 1▼ | +0 % [−3 ; +1] 1▼ | +3 % [+1 ; +8] 1▼ | +0,01 [−0,01 ; +0,02] | −3 [−20 ; +1] 2▲ | +0,006 [−0,001 ; +0,142] 3▼ | +0,002 [−0,041 ; +0,137] 1▲ 1▼ | +0,02 [−0,96 ; +0,08] 1▼ |
| V2-C | +0,10 [−0,15 ; +0,16] 5▲ 1▼ | +0,12 [+0,04 ; +0,20] 5▲ | +0,06 [+0,02 ; +0,12] 4▲ | +8 % [−1 ; +17] 5▲ | +2 % [−3 ; +3] 1▲ | −0 % [−2 ; +8] 2▼ | +0,01 [−0,02 ; +0,06] 1▲ | −15 [−20 ; −3] 4▲ | +0,034 [−0,004 ; +0,142] 4▼ | +0,009 [−0,046 ; +0,212] 1▲ 3▼ | +0,08 [−0,96 ; +0,89] 3▲ 2▼ |
<!-- /crossed -->

**Dans l'ensemble** : ce que chaque changement apporte quand les cinq autres sont là (V2 face à V2 sans lui) :

<!-- crossed-within:express -->
| | Profondeur | Bruit fin | Bruit large | S/B cible | Étoiles faibles | FWHM | Rondeur | Points 1 couleur | Couleur B−R | Couleur V−R | Planéité |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 (V2 − V2-S1) | +0,03 [−0,12 ; +0,15] 3▲ 1▼ | +0,06 [+0,02 ; +0,19] 3▲ | +0,00 [−0,02 ; +0,21] 2▲ | +5 % [+1 ; +21] 3▲ | +0 % [−2 ; +9] 2▲ | +4 % [+1 ; +4] 4▼ | +0,00 [−0,01 ; +0,02] | −6 [−18 ; +3] 3▲ | +0,002 [−0,001 ; +0,008] 1▼ | +0,005 [−0,032 ; +0,020] 2▲ 1▼ | +0,06 [−0,13 ; +0,11] 1▲ 1▼ |
| S2 (V2 − V2-S2) | −0,02 [−0,07 ; −0,01] 1▼ | −0,01 [−0,02 ; −0,00] | −0,01 [−0,02 ; −0,00] | +0 % [−2 ; +1] | −1 % [−2 ; +0] | +0 % [−0 ; +1] | −0,00 [−0,00 ; +0,00] | +0 [−1 ; +2] | +0,003 [−0,000 ; +0,049] 2▼ | +0,001 [−0,003 ; +0,009] | −0,00 [−0,01 ; +0,02] |
| S3 (V2 − V2-S3) | +0,00 [−0,14 ; +0,03] 1▼ | +0,00 [−0,00 ; +0,01] | +0,00 [−0,01 ; +0,01] | +0 % [−1 ; +0] | −1 % [−4 ; +3] 1▲ 1▼ | +0 % [+0 ; +1] | +0,00 [−0,01 ; +0,00] | +3 [−1 ; +3] | +0,001 [−0,004 ; +0,010] 1▼ | +0,000 [−0,010 ; +0,070] 1▲ 1▼ | −0,00 [−0,43 ; +0,01] 1▼ |
| S4 (V2 − V2-S4) | −0,00 [−0,05 ; +0,02] 2▼ | −0,00 [−0,03 ; +0,03] | −0,00 [−0,06 ; +0,03] 1▼ | −1 % [−6 ; +4] 1▲ 1▼ | +0 % [−2 ; +1] | −1 % [−1 ; +0] | +0,00 [−0,00 ; +0,02] | +0 [−5 ; +3] 1▲ | +0,000 [−0,002 ; +0,004] | −0,000 [−0,122 ; +0,001] 2▲ | −0,01 [−0,06 ; +0,03] |
| S5 (V2 − V2-S5) | −0,02 [−0,05 ; +0,05] 1▲ 2▼ | −0,00 [−0,00 ; +0,01] | +0,00 [−0,00 ; +0,00] | −0 % [−11 ; +1] 1▼ | −0 % [−2 ; +1] | −0 % [−0 ; +1] | +0,00 [−0,00 ; +0,00] | +1 [−2 ; +2] | +0,000 [−0,000 ; +0,032] 1▼ | +0,001 [−0,012 ; +0,005] 1▲ | −0,01 [−0,05 ; +0,01] |
| S6 (V2 − V2-S6) | −0,02 [−0,03 ; −0,00] | +0,00 [−0,00 ; +0,00] | −0,00 [−0,01 ; +0,00] | +0 % [−1 ; +1] | −0 % [−2 ; +3] | −0 % [−0 ; +0] | −0,00 [−0,00 ; +0,00] | +1 [−1 ; +2] | −0,000 [−0,016 ; +0,013] 1▲ 1▼ | −0,001 [−0,006 ; +0,098] 1▼ | −0,00 [−0,48 ; +0,01] 1▼ |
<!-- /crossed -->

En clair, changement par changement :

- **OLD-bis (le témoin)** : un recalage à peine différent bouge la profondeur de 0,03 mag au plus, le reste de
  presque rien. Les seuils du critère (0,04 mag, 3 %) sont donc honnêtes.
- **S1, FWHM de Moffat + règle de couverture** : le seul changement qui gagne seul sans rien casser d'important. La
  règle choisit partout l'échelle 1 avec des gouttes d'un pixel entier (au lieu de 0,7) : le bruit fin baisse de
  0,02 à 0,04 mag et le signal/bruit sur la cible monte de 2 à 6 %, au prix d'étoiles 1 à 4 % plus larges (Heart : 4 %,
  au-delà du seuil). La FWHM de Moffat mesure enfin la vraie taille des étoiles (Heart 1,7 px, l'ancienne mesure lisait
  près de 4 px) ; c'est elle qui rend la sélection k × σ utilisable (voir S4). Effet sur les couleurs : V−R un peu
  moins juste sur le Seestar et la Rosette (+0,03).
- **S2, rejet de Hubble** : il rejette 0,004 à 0,22 % des échantillons (OLD : 0,001 à 0,04 % ; Siril : 0,8 à 1,2 %),
  sans gain mesurable ; il laisse un peu plus de points d'une seule couleur (M51, Rosette) et coûte 0,07 mag sur M51.
  Dans l'ensemble V2, l'enlever ne change presque rien.
- **S3, pondération par le bruit** : neutre seul (±0,02 mag). Mais avec la FWHM de Moffat et la goutte de 1 (S1), il
  coûte 0,14 mag de profondeur et 4 % d'étoiles faibles sur Iris : c'est ce couple qui fait perdre V2 sur Iris.
- **S4, sélection k × σ** : en Express, sans la FWHM de Moffat, elle écarte des poses sur la rondeur mesurée en blocs
  de 2 × 2 (M51 : 7 poses sur 30, M8 : 5 sur 15, Seestar : 6 sur 30) et perd jusqu'à 0,22 mag. Sur la Rosette, elle
  garde 23 poses au lieu de 16 (OLD écartait les poses nuageuses). Avec Moffat (dans V2), elle n'écarte presque plus
  rien (0 à 2 poses, 7 sur la Rosette). **S4b = S4 partout** : le filtre de transparence n'a écarté aucune pose, sur
  aucun cas.
- **S5, normalisation locale** : neutre partout (sauf −11 % de signal/bruit sur la galaxie de M51).
- **S6, recalage des couleurs par blocs** : neutre sur toutes les mesures, y compris les couleurs des étoiles
  (Seestar : B−R +0,03). Le décalage mesuré est petit sur ces cas (rouge 0,1 à 0,4 px au centre).
- **S7, méthode Classique** : de loin le changement le plus fort. Profondeur +0,11 à +0,17 mag sur Heart, Iris, le
  Seestar et la Rosette (M51 égale, M8 −0,13), bruit fin +0,04 à +0,18, signal/bruit +8 à +11 %, points d'une seule
  couleur au niveau de Siril, taux de rejet comme Siril (0,5 à 1,3 %). Son bruit fin est celui de Siril à 0,02 mag près
  sur les six cas : le module refait bien la méthode de Siril. Mais elle **abîme les couleurs** : erreur B−R et V−R
  plus forte sur Heart, Iris et M8 (Heart : 0,09 → 0,23, le niveau de Siril ; M8 en V−R : 0,05 → 0,28), une dominante
  orangée sur Heart (le blanc passe de BP−RP 0,88 à 0,71) avec un fond bien moins plat (−1 mag ; M8 −0,3), et des
  étoiles 8 % plus larges sur Heart (sous-échantillonné : le dématriçage les élargit), 5 % sur M8.
- **V2, tout ensemble en drizzle** : pas de gain net. +0,02 de bruit fin, +1 % de signal/bruit, étoiles 3 % plus
  larges ; Iris −0,13 mag (le couple S1 + S3), M51 −0,05 mag et −8 % de signal/bruit. **V2-auto** ne diffère que
  sur Heart (la règle y conseille la Classique, avec ses défauts de couleur) ; **V2-C** (tout, en Classique) se
  comporte comme S7.

### La configuration retenue

Le critère, appliqué aux 65 cases de chaque configuration :

| Configuration | Cases moins bien qu'OLD | Cases mieux qu'OLD | Cases mieux que Siril | Cases moins bien que Siril |
|---|---|---|---|---|
| OLD-bis | 0 | 0 | 15 | 31 |
| S1 | 3 | 7 | 17 | 32 |
| S2 | 5 | 2 | 14 | 34 |
| S3 | 4 | 4 | 15 | 32 |
| S4 | 16 | 8 | 15 | 31 |
| S4b | 16 | 8 | 15 | 31 |
| S5 | 3 | 0 | 15 | 32 |
| S6 | 1 | 0 | 14 | 30 |
| S7 | 11 | 32 | 7 | 14 |
| V2 | 9 | 5 | 16 | 33 |
| V2-auto | 10 | 9 | 12 | 31 |
| V2-C | 12 | 29 | 9 | 14 |
| V2-S1 | 18 | 7 | 12 | 33 |
| V2-S2 | 4 | 10 | 18 | 29 |
| V2-S3 | 5 | 10 | 18 | 29 |
| V2-S4 | 12 | 6 | 15 | 35 |
| V2-S5 | 5 | 9 | 17 | 32 |
| V2-S6 | 7 | 9 | 16 | 31 |

- **Aucune configuration ne passe sans aucune case « moins bien ».** S6 seul n'en a qu'une, S5 seul trois, mais ils ne
  font mieux qu'OLD nulle part : ils ne changent rien de mesurable, ce ne sont pas des candidats.
- **S1 (FWHM de Moffat + règle de couverture) est retenu** : le moins de cases moins bien (3) parmi les configurations
  qui apportent quelque chose (7 cases mieux qu'OLD), et 17 cases mieux que Siril. Ses trois défauts sont petits : des
  étoiles de Heart 4 % plus larges (la goutte d'un pixel entier), et l'erreur de couleur V−R du Seestar et de la
  Rosette +0,03.
- Juste derrière : **V2-S2** (tout sauf le rejet de Hubble) et **V2-S3** (tout sauf la pondération par le bruit), 4 et
  5 cases moins bien, mais 10 cases mieux qu'OLD et 18 mieux que Siril. Ils ajoutent à S1 des changements neutres ici
  (recalage des couleurs par blocs, normalisation locale) ; tous deux perdent sur la cible de M51 et la finesse de
  Heart, V2-S2 aussi sur la profondeur d'Iris, V2-S3 sur les couleurs B−R.
- **La Classique (S7, V2-C)** est la seule qui rejoint Siril sur le bruit et la profondeur (32 cases mieux qu'OLD, 7
  seulement mieux que Siril mais aussi 14 seulement moins bien), mais avec 11 à 12 cases moins bien qu'OLD, presque
  toutes des couleurs (Heart, Iris, M8) et la planéité de Heart. Elle ne passe pas le critère tant que ses couleurs
  ne sont pas réglées (voir « Ce qui reste »).

**S1 en Express, cas par cas** (valeur · écart à OLD · écart à Siril ; ▲ mieux, ▼ moins bien, au-delà du bruit) :

<!-- crossed-case:S1:express -->
| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Profondeur | 16,25 · OLD +0,03 · Siril −0,11 ▼ | 17,29 · OLD +0,00 · Siril −0,15 ▼ | 18,24 · OLD +0,01 · Siril −0,00 | 17,90 · OLD +0,01 · Siril +0,16 ▲ | 15,94 · OLD +0,03 · Siril −0,10 ▼ | 17,07 · OLD +0,01 · Siril −0,27 ▼ |
| Bruit fin | 24,50 · OLD +0,03 · Siril −0,10 ▼ | 23,73 · OLD +0,04 ▲ · Siril −0,13 ▼ | 24,57 · OLD +0,04 · Siril −0,07 ▼ | 24,07 · OLD +0,02 · Siril −0,02 | 22,54 · OLD +0,03 · Siril −0,08 ▼ | 23,51 · OLD +0,04 ▲ · Siril −0,11 ▼ |
| Bruit large | 25,25 · OLD +0,00 · Siril −0,04 ▼ | 25,12 · OLD +0,02 · Siril −0,11 ▼ | 25,89 · OLD +0,02 · Siril +0,02 | 24,36 · OLD +0,01 · Siril −0,02 | 23,71 · OLD +0,02 · Siril −0,08 ▼ | 24,75 · OLD +0,01 · Siril −0,06 ▼ |
| S/B cible | 4,7 · OLD +5 % ▲ · Siril −8 % ▼ | 2,4 · OLD +2 % · Siril −20 % ▼ | 4,2 · OLD +4 % ▲ · Siril −9 % ▼ | 46,4 · OLD +2 % · Siril −5 % ▼ | 24,1 · OLD +6 % ▲ · Siril +9 % ▲ | 4,3 · OLD +3 % ▲ · Siril −12 % ▼ |
| Étoiles faibles | 21892 · OLD −0 % · Siril −1 % | 4597 · OLD −0 % · Siril −4 % ▼ | 1219 · OLD +1 % · Siril −0 % | 27709 · OLD +0 % · Siril +3 % | 552 · OLD +1 % · Siril −3 % ▼ | 5218 · OLD −1 % · Siril −5 % ▼ |
| FWHM | 12,56 · OLD +4 % ▼ · Siril −4 % ▲ | 6,24 · OLD +2 % · Siril +2 % | 6,99 · OLD +1 % · Siril +1 % | 5,44 · OLD +2 % · Siril −3 % ▲ | 6,12 · OLD +3 % · Siril +5 % ▼ | 6,05 · OLD +1 % · Siril +4 % ▼ |
| Rondeur | 0,91 · OLD +0,01 · Siril +0,00 | 0,88 · OLD +0,00 · Siril −0,00 | 0,84 · OLD +0,00 · Siril +0,01 | 0,95 · OLD +0,00 · Siril +0,02 | 0,87 · OLD +0,01 · Siril −0,02 | 0,76 · OLD +0,01 · Siril +0,00 |
| Points 1 couleur | 24 · OLD +3 · Siril +23 ▼ | 21 · OLD −4 ▲ · Siril +18 ▼ | 17 · OLD −0 · Siril +13 ▼ | 2472 · OLD +149 · Siril +1054 ▼ | 8 · OLD +3 · Siril +8 ▼ | 22 · OLD +0 · Siril +15 ▼ |
| Couleur B−R | 0,094 · OLD +0,001 · Siril −0,124 ▲ | 0,045 · OLD −0,001 · Siril −0,013 ▲ | 0,026 · OLD −0,000 · Siril +0,005 | 0,056 · OLD +0,003 · Siril −0,018 ▲ | 0,113 · OLD +0,001 · Siril −0,035 ▲ | 0,062 · OLD +0,002 · Siril +0,008 ▼ |
| Couleur V−R | 0,184 · OLD −0,002 · Siril −0,141 ▲ | 0,057 · OLD −0,002 · Siril −0,015 ▲ | 0,035 · OLD −0,002 · Siril −0,012 ▲ | 0,049 · OLD −0,003 · Siril −0,078 ▲ | 0,181 · OLD +0,025 ▼ · Siril −0,071 ▲ | 0,238 · OLD +0,030 ▼ · Siril +0,131 ▼ |
| Planéité | 24,71 · OLD +0,06 · Siril +0,13 ▲ | 23,45 · OLD +0,07 · Siril +0,19 ▲ | 23,15 · OLD −0,05 · Siril +0,17 ▲ | 23,26 · OLD +0,06 · Siril +0,10 ▲ | 21,04 · OLD +0,08 · Siril −0,26 ▼ | 21,23 · OLD +0,01 · Siril −0,71 ▼ |
| Traînées (dérive) | 1,00 · OLD +0,00 · Siril +0,00 | 0,99 · OLD +0,01 · Siril −0,01 | 1,04 · OLD +0,00 · Siril +0,03 | 0,99 · OLD +0,00 · Siril −0,01 | 0,95 · OLD −0,01 · Siril −0,01 | 1,00 · OLD −0,00 · Siril −0,00 |
| Poses | 30 · OLD +0 · Siril +0 | 30 · OLD +0 · Siril +0 | 30 · OLD +0 · Siril +0 | 15 · OLD +0 · Siril +0 | 30 · OLD +0 · Siril +1 | 16 · OLD +0 · Siril −14 |
| Rejet % | 0,014 · OLD +0,000 · Siril −0,927 | 0,045 · OLD +0,001 · Siril −0,847 | 0,025 · OLD +0,001 · Siril −0,779 | 0,014 · OLD +0,000 · Siril −1,208 | 0,001 · OLD +0,000 · Siril −0,918 | 0,038 · OLD −0,001 · Siril −0,872 |
| Empilement (min) | 1,9 · OLD +1,2 · Siril – | 0,9 · OLD +0,2 · Siril – | 1,5 · OLD +0,2 · Siril – | 5,0 · OLD +1,3 · Siril – | 0,2 · OLD +0,0 · Siril – | 0,9 · OLD +0,3 · Siril – |
<!-- /crossed -->

**V2 (tous les changements, drizzle) en Express**, pour comparaison :

<!-- crossed-case:V2:express -->
| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Profondeur | 16,22 · OLD +0,00 · Siril −0,13 ▼ | 17,15 · OLD −0,13 ▼ · Siril −0,29 ▼ | 18,17 · OLD −0,05 ▼ · Siril −0,06 ▼ | 17,91 · OLD +0,02 · Siril +0,17 ▲ | 15,93 · OLD +0,01 · Siril −0,11 ▼ | 17,07 · OLD +0,01 · Siril −0,27 ▼ |
| Bruit fin | 24,50 · OLD +0,02 · Siril −0,10 ▼ | 23,69 · OLD +0,00 · Siril −0,17 ▼ | 24,56 · OLD +0,03 · Siril −0,08 ▼ | 24,05 · OLD +0,01 · Siril −0,03 | 22,51 · OLD +0,00 · Siril −0,11 ▼ | 23,54 · OLD +0,07 ▲ · Siril −0,08 ▼ |
| Bruit large | 25,24 · OLD −0,00 · Siril −0,05 ▼ | 25,08 · OLD −0,02 · Siril −0,14 ▼ | 25,89 · OLD +0,01 · Siril +0,02 | 24,35 · OLD +0,00 · Siril −0,03 | 23,67 · OLD −0,02 · Siril −0,12 ▼ | 24,76 · OLD +0,03 · Siril −0,05 ▼ |
| S/B cible | 4,7 · OLD +6 % ▲ · Siril −7 % ▼ | 2,4 · OLD +1 % · Siril −20 % ▼ | 3,8 · OLD −8 % ▼ · Siril −19 % ▼ | 45,9 · OLD +1 % · Siril −6 % ▼ | 22,9 · OLD +0 % · Siril +4 % ▲ | 4,6 · OLD +9 % ▲ · Siril −7 % ▼ |
| Étoiles faibles | 21820 · OLD −0 % · Siril −2 % | 4461 · OLD −3 % ▼ · Siril −7 % ▼ | 1211 · OLD +0 % · Siril −1 % | 27732 · OLD +0 % · Siril +3 % | 555 · OLD +1 % · Siril −3 % | 5217 · OLD −1 % · Siril −5 % ▼ |
| FWHM | 12,69 · OLD +5 % ▼ · Siril −3 % ▲ | 6,30 · OLD +3 % · Siril +3 % ▼ | 7,05 · OLD +2 % · Siril +2 % | 5,46 · OLD +3 % · Siril −3 % | 6,10 · OLD +3 % · Siril +4 % ▼ | 6,01 · OLD +1 % · Siril +4 % ▼ |
| Rondeur | 0,91 · OLD +0,01 · Siril +0,00 | 0,89 · OLD +0,01 · Siril +0,00 | 0,83 · OLD −0,01 · Siril +0,00 | 0,95 · OLD −0,00 · Siril +0,01 | 0,87 · OLD +0,01 · Siril −0,01 | 0,77 · OLD +0,01 · Siril +0,01 |
| Points 1 couleur | 26 · OLD +5 ▼ · Siril +25 ▼ | 22 · OLD −3 · Siril +19 ▼ | 19 · OLD +1 · Siril +14 ▼ | 2528 · OLD +205 · Siril +1110 ▼ | 5 · OLD −0 · Siril +5 ▼ | 17 · OLD −4 ▲ · Siril +10 ▼ |
| Couleur B−R | 0,100 · OLD +0,007 ▼ · Siril −0,118 ▲ | 0,050 · OLD +0,005 · Siril −0,007 ▲ | 0,028 · OLD +0,002 · Siril +0,007 ▼ | 0,061 · OLD +0,007 ▼ · Siril −0,013 ▲ | 0,124 · OLD +0,012 ▼ · Siril −0,024 ▲ | 0,059 · OLD −0,001 · Siril +0,005 |
| Couleur V−R | 0,184 · OLD −0,002 · Siril −0,142 ▲ | 0,059 · OLD +0,000 · Siril −0,013 ▲ | 0,033 · OLD −0,004 · Siril −0,014 ▲ | 0,059 · OLD +0,007 · Siril −0,068 ▲ | 0,115 · OLD −0,041 ▲ · Siril −0,137 ▲ | 0,212 · OLD +0,005 · Siril +0,106 ▼ |
| Planéité | 24,70 · OLD +0,05 · Siril +0,11 ▲ | 23,41 · OLD +0,03 · Siril +0,15 ▲ | 23,12 · OLD −0,08 · Siril +0,14 ▲ | 23,28 · OLD +0,08 · Siril +0,12 ▲ | 21,03 · OLD +0,07 · Siril −0,26 ▼ | 21,23 · OLD +0,01 · Siril −0,72 ▼ |
| Traînées (dérive) | 1,00 · OLD −0,00 · Siril −0,00 | 0,99 · OLD +0,01 · Siril −0,01 | 1,04 · OLD −0,00 · Siril +0,03 | 0,99 · OLD +0,00 · Siril −0,01 | 0,95 · OLD −0,01 · Siril −0,01 | 1,01 · OLD +0,01 · Siril +0,01 |
| Poses | 30 · OLD +0 · Siril +0 | 28 · OLD −2 · Siril −2 | 30 · OLD +0 · Siril +0 | 14 · OLD −1 · Siril −1 | 28 · OLD −2 · Siril −1 | 23 · OLD +7 · Siril −7 |
| Rejet % | 0,020 · OLD +0,006 · Siril −0,921 | 0,145 · OLD +0,101 · Siril −0,747 | 0,046 · OLD +0,022 · Siril −0,758 | 0,013 · OLD −0,001 · Siril −1,209 | 0,004 · OLD +0,003 · Siril −0,915 | 0,173 · OLD +0,134 · Siril −0,737 |
| Empilement (min) | 1,3 · OLD +0,5 · Siril – | 0,8 · OLD +0,1 · Siril – | 2,9 · OLD +1,6 · Siril – | 4,2 · OLD +0,5 · Siril – | 0,2 · OLD +0,1 · Siril – | 1,2 · OLD +0,6 · Siril – |
<!-- /crossed -->

### En Base (100 poses)

OLD, S1 et V2 sur les 100 poses de chaque cas (M8 n'a que ses 15 poses). Avec 100 poses, la règle de couverture
de S1 garde l'échelle 1,5 (goutte 0,7) pour les petites étoiles de Heart (1,7 px) et du Seestar (1,9 px) : **S1 y
donne exactement l'image d'OLD**. Elle passe à l'échelle 1 (goutte 1) pour Iris (3,1 px) et la Rosette (2,7 px), où
OLD prenait 1,5.

- **S1 face à OLD** : Iris +0,02 mag, bruit à grande échelle +0,06, signal/bruit +4 %, fond plus plat (+0,2), étoiles
  3 % plus larges ; M51 +0,03 mag ; **la Rosette perd** : −0,08 mag, étoiles 4 % plus larges, couleurs des étoiles
  moins justes (V−R 0,05 → 0,14) et un ciel qui tire au magenta sur la planche, fond moins plat (−0,34). En Express, la
  Rosette était déjà à l'échelle 1 et ses couleurs V−R déjà moins bonnes que chez Siril : l'échelle 1 ne lui réussit
  pas. Au total 6 cases moins bien qu'OLD (5 sur la Rosette) et 6 mieux.
- **V2 face à OLD** : sa sélection écarte maintenant des poses (Heart 5, dont 3 pour la transparence ; M51 9 et le
  Seestar 12 pour la FWHM) et le signal/bruit baisse d'autant (M51 −10 %, Seestar −13 %) ; sur la Rosette elle garde
  90 poses au lieu de 56 (+12 % de signal/bruit, bruit fin +0,09) mais avec les défauts de couleur de l'échelle 1.
  11 cases moins bien, 6 mieux.
- **Les durées** : en Base, calibration + alignement + empilement prennent 3,7 à 6,4 min chez nous contre 4,2 à 9,1 min
  pour la préparation de Siril (le Seestar mis à part).

**S1 en Base, cas par cas :**

<!-- crossed-case:S1:base -->
| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Profondeur | 16,69 · OLD +0,03 · Siril +0,08 ▲ | 17,96 · OLD +0,02 · Siril −0,13 ▼ | 18,75 · OLD +0,03 · Siril −0,04 | 16,70 · OLD +0,00 · Siril +0,14 ▲ | 17,60 · OLD −0,08 ▼ · Siril −0,12 ▼ |
| Bruit fin | 24,73 · OLD +0,00 · Siril −0,09 ▼ | 24,39 · OLD +0,02 · Siril −0,15 ▼ | 25,22 · OLD +0,03 · Siril −0,08 ▼ | 23,15 · OLD +0,00 · Siril −0,05 ▼ | 24,11 · OLD +0,02 · Siril −0,13 ▼ |
| Bruit large | 25,35 · OLD +0,00 · Siril +0,06 ▲ | 25,71 · OLD +0,06 ▲ · Siril −0,14 ▼ | 26,46 · OLD +0,01 · Siril −0,03 | 24,34 · OLD +0,00 · Siril +0,09 ▲ | 25,19 · OLD +0,07 ▲ · Siril −0,08 ▼ |
| S/B cible | 6,3 · OLD +0 % · Siril −2 % | 3,7 · OLD +4 % ▲ · Siril −12 % ▼ | 3,1 · OLD +3 % · Siril −10 % ▼ | 36,6 · OLD +0 % · Siril −0 % | 7,5 · OLD +3 % · Siril −9 % ▼ |
| Étoiles faibles | 24718 · OLD +0 % · Siril +1 % | 6416 · OLD −0 % · Siril −3 % | 1567 · OLD +1 % · Siril −4 % ▼ | 1065 · OLD +0 % · Siril +3 % ▲ | 6695 · OLD −1 % · Siril −2 % |
| FWHM | 11,65 · OLD +0 % · Siril −12 % ▲ | 6,16 · OLD +3 % ▼ · Siril +2 % | 7,09 · OLD +1 % · Siril +1 % | 5,70 · OLD +0 % · Siril −0 % | 5,91 · OLD +4 % ▼ · Siril +6 % ▼ |
| Rondeur | 0,88 · OLD −0,00 · Siril −0,04 ▼ | 0,89 · OLD +0,01 · Siril −0,00 | 0,83 · OLD +0,00 · Siril +0,01 | 0,87 · OLD +0,00 · Siril −0,05 ▼ | 0,79 · OLD +0,02 · Siril +0,01 |
| Points 1 couleur | 125 · OLD −1 · Siril +125 ▼ | 15 · OLD −4 ▲ · Siril +11 ▼ | 26 · OLD −1 · Siril +21 ▼ | 9 · OLD +0 · Siril +9 ▼ | 10 · OLD −8 ▲ · Siril +7 ▼ |
| Couleur B−R | 0,063 · OLD +0,000 · Siril −0,110 ▲ | 0,038 · OLD +0,000 · Siril −0,011 ▲ | 0,027 · OLD +0,001 · Siril +0,009 ▼ | 0,048 · OLD +0,000 · Siril −0,027 ▲ | 0,046 · OLD +0,012 ▼ · Siril +0,005 |
| Couleur V−R | 0,100 · OLD +0,000 · Siril −0,155 ▲ | 0,039 · OLD +0,000 · Siril −0,011 ▲ | 0,028 · OLD −0,000 · Siril −0,012 ▲ | 0,056 · OLD +0,000 · Siril −0,170 ▲ | 0,135 · OLD +0,085 ▼ · Siril +0,046 ▼ |
| Planéité | 24,80 · OLD −0,00 · Siril +0,07 | 23,62 · OLD +0,20 ▲ · Siril −0,03 | 23,18 · OLD +0,01 · Siril +0,18 ▲ | 21,25 · OLD +0,00 · Siril −0,26 ▼ | 21,52 · OLD −0,34 ▼ · Siril −0,48 ▼ |
| Traînées (dérive) | 1,00 · OLD −0,00 · Siril +0,00 | 0,98 · OLD −0,01 · Siril −0,03 ▲ | 1,04 · OLD +0,00 · Siril −0,01 | 1,05 · OLD +0,00 · Siril +0,01 | 1,05 · OLD +0,01 · Siril +0,01 |
| Poses | 99 · OLD +0 · Siril −1 | 99 · OLD +0 · Siril −1 | 100 · OLD +0 · Siril +0 | 100 · OLD +0 · Siril +7 | 56 · OLD +0 · Siril −44 |
| Rejet % | 0,021 · OLD +0,000 · Siril −0,739 | 0,053 · OLD +0,003 · Siril −0,669 | 0,030 · OLD +0,000 · Siril −0,599 | 0,001 · OLD +0,000 · Siril −0,717 | 0,047 · OLD +0,005 · Siril −0,691 |
| Empilement (min) | 2,5 · OLD −0,4 · Siril – | 2,1 · OLD −0,7 · Siril – | 4,5 · OLD +0,6 · Siril – | 0,6 · OLD −3,5 · Siril – | 1,2 · OLD −0,6 · Siril – |
<!-- /crossed -->

**V2 en Base :**

<!-- crossed-case:V2:base -->
| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Profondeur | 16,65 · OLD −0,01 · Siril +0,04 | 17,96 · OLD +0,03 · Siril −0,12 ▼ | 18,73 · OLD +0,01 · Siril −0,06 ▼ | 16,66 · OLD −0,04 · Siril +0,10 ▲ | 17,66 · OLD −0,02 · Siril −0,05 ▼ |
| Bruit fin | 24,72 · OLD −0,00 · Siril −0,09 ▼ | 24,38 · OLD +0,01 · Siril −0,16 ▼ | 25,18 · OLD −0,02 · Siril −0,12 ▼ | 23,09 · OLD −0,06 ▼ · Siril −0,11 ▼ | 24,19 · OLD +0,09 ▲ · Siril −0,05 ▼ |
| Bruit large | 25,33 · OLD −0,01 · Siril +0,05 ▲ | 25,71 · OLD +0,06 ▲ · Siril −0,14 ▼ | 26,42 · OLD −0,03 · Siril −0,07 ▼ | 24,35 · OLD +0,01 · Siril +0,09 ▲ | 25,25 · OLD +0,13 ▲ · Siril −0,02 |
| S/B cible | 6,1 · OLD −3 % ▼ · Siril −5 % ▼ | 3,8 · OLD +8 % ▲ · Siril −8 % ▼ | 2,7 · OLD −10 % ▼ · Siril −21 % ▼ | 31,8 · OLD −13 % ▼ · Siril −13 % ▼ | 8,2 · OLD +12 % ▲ · Siril −1 % |
| Étoiles faibles | 24613 · OLD −0 % · Siril +1 % | 6418 · OLD −0 % · Siril −3 % | 1559 · OLD +0 % · Siril −4 % ▼ | 1048 · OLD −2 % · Siril +2 % | 6809 · OLD +1 % · Siril +0 % |
| FWHM | 11,75 · OLD +1 % · Siril −12 % ▲ | 6,21 · OLD +4 % ▼ · Siril +3 % | 7,06 · OLD +0 % · Siril +1 % | 5,61 · OLD −1 % · Siril −2 % | 5,87 · OLD +3 % ▼ · Siril +5 % ▼ |
| Rondeur | 0,88 · OLD −0,00 · Siril −0,04 ▼ | 0,89 · OLD +0,01 · Siril −0,01 | 0,83 · OLD +0,01 · Siril +0,02 | 0,89 · OLD +0,02 · Siril −0,03 ▼ | 0,78 · OLD +0,02 · Siril +0,01 |
| Points 1 couleur | 144 · OLD +18 · Siril +143 ▼ | 20 · OLD +1 · Siril +16 ▼ | 23 · OLD −4 · Siril +19 ▼ | 13 · OLD +4 ▼ · Siril +13 ▼ | 18 · OLD +1 · Siril +15 ▼ |
| Couleur B−R | 0,064 · OLD +0,002 · Siril −0,109 ▲ | 0,038 · OLD +0,001 · Siril −0,011 ▲ | 0,026 · OLD +0,000 · Siril +0,009 ▼ | 0,054 · OLD +0,006 ▼ · Siril −0,021 ▲ | 0,045 · OLD +0,012 ▼ · Siril +0,004 |
| Couleur V−R | 0,098 · OLD −0,002 · Siril −0,156 ▲ | 0,040 · OLD +0,001 · Siril −0,010 ▲ | 0,027 · OLD −0,001 · Siril −0,013 ▲ | 0,061 · OLD +0,005 · Siril −0,165 ▲ | 0,132 · OLD +0,082 ▼ · Siril +0,043 ▼ |
| Planéité | 24,81 · OLD +0,01 · Siril +0,08 | 23,62 · OLD +0,20 ▲ · Siril −0,03 | 23,20 · OLD +0,03 · Siril +0,20 ▲ | 21,24 · OLD −0,01 · Siril −0,27 ▼ | 21,59 · OLD −0,28 ▼ · Siril −0,41 ▼ |
| Traînées (dérive) | 1,00 · OLD −0,00 · Siril −0,00 | 0,99 · OLD −0,00 · Siril −0,02 | 1,03 · OLD −0,01 · Siril −0,03 | 1,04 · OLD −0,01 · Siril −0,00 | 1,06 · OLD +0,03 · Siril +0,02 |
| Poses | 94 · OLD −5 · Siril −6 | 97 · OLD −2 · Siril −3 | 91 · OLD −9 · Siril −9 | 88 · OLD −12 · Siril −5 | 90 · OLD +34 · Siril −10 |
| Rejet % | 0,014 · OLD −0,007 · Siril −0,746 | 0,154 · OLD +0,104 · Siril −0,568 | 0,050 · OLD +0,020 · Siril −0,579 | 0,004 · OLD +0,003 · Siril −0,714 | 0,134 · OLD +0,092 · Siril −0,604 |
| Empilement (min) | 2,0 · OLD −0,8 · Siril – | 2,0 · OLD −0,8 · Siril – | 2,9 · OLD −1,0 · Siril – | 0,6 · OLD −3,5 · Siril – | 1,9 · OLD +0,1 · Siril – |
<!-- /crossed -->

### Ce qui reste moins bien que Siril

Avec S1, la configuration retenue :

1. **La profondeur et le bruit fin.** En Express, Siril va 0,10 à 0,27 mag plus profond sur Heart, Iris, le Seestar
   et la Rosette (M51 égal, M8 devant de 0,16) ; en Base, Iris −0,13, Rosette −0,12, M51 −0,04 (Heart et Seestar
   devant). En Base, le bruit fin reste 0,05 à 0,15 mag derrière sur les cinq cas, le signal/bruit sur la cible 2 à
   12 % derrière (5 à 20 % en Express ; le Seestar à part). Seule la méthode Classique comble cet écart sur nos
   images, au prix des couleurs.
2. **Les points d'une seule couleur** : 8 à 24 par degré² en Express, 9 à 125 en Base (Heart), contre 0 à 5 chez
   Siril. Aucun changement v2 en drizzle ne les fait baisser ; la Classique, oui (au niveau de Siril).
3. **La Rosette à l'échelle 1** : couleurs des étoiles (V−R) et planéité du fond derrière Siril, en Express comme en
   Base avec S1 et V2.
4. **La planéité du Seestar** (−0,26 mag), inchangée par tous les changements.
5. **La méthode Classique**, la seule qui rejoint Siril sur le bruit, donne à Heart une dominante orangée (le blanc
   passe de BP−RP 0,88 à 0,71, erreur de couleur 0,09 → 0,23) et un fond moins plat, et des couleurs moins justes sur
   Iris et M8. Les couleurs de notre étape Couleur sur une pile dématriçée sont à regarder avant de s'en servir.

**Défauts du moteur trouvés pendant le banc** (rien n'a été corrigé ni réglé dans le moteur) :

- **Encodage du journal** : le processus de calcul écrit son journal sur une sortie ouverte avec l'encodage de
  Windows (cp1252) ; un message contenant « σ » lève `UnicodeEncodeError` dans `core/journal.py` (ligne 42, le `print`)
  et l'étape échoue. Touché : toute configuration avec la normalisation locale (`stages/stack.py` ligne 369, le message
  « normalisation (local): … σ ») et la méthode Classique (son message final « 5 / 3,5 σ », ligne 1161). Le client
  lance ses calculs de la même façon : il y est exposé. Le banc l'évite avec `PYTHONIOENCODING=utf-8`.
- **Méthode Classique et CUDA** : sur M8 (14 poses, donc le rejet « winsorized »), une assertion CUDA
  (`TensorCompare.cu:109`, `input[0] != 0`) est remontée dans `core/rejection_classic.py` ligne 180
  (`winsorized_clip`) ; le contexte CUDA du processus est alors perdu et **toutes les étapes suivantes du calcul
  échouent** (8 empilements d'un coup). Relancées seules, S7 et V2-C sont passées : l'erreur n'est pas systématique.
  Une autre fois, « unspecified launch failure » dans une Classique (Rosette V2-C, rejet « linear fit ») pendant qu'un
  autre empilement tournait sur la même carte ; relancée seule, elle est passée.
- **La règle de couverture juge avant la sélection** : sur la Rosette (Express), elle estime la couverture sur les
  30 poses alignées (19 % : drizzle conseillé), alors que la sélection n'en garde que 23, dont la couverture mesurée
  varie de 19 % (rouge) à 27 % (bleu), au-delà des 20 % de la règle. Le journal du pilote annonçait « Classique pour
  Heart et la Rosette » ; la règle du moteur ne la conseille plus que pour Heart.
- Hors moteur : la machine était près de sa limite de mémoire (Photoshop en tenait 66 Go) ; deux étapes ont échoué
  en `MemoryError` et ont été relancées.

### Les planches

Siril à gauche, puis nos configurations, sur la même zone du ciel. Ligne 1 : les images finales (rendus par défaut ;
Siril éclaircit le fond, nous le gardons sombre). Ligne 2 : les piles linéaires étirées par leur propre bruit. Lignes
3 et 4 : la cible et un coin à 100 %. Ligne 5 : un coin de ciel de la pile linéaire, chaque couleur étirée par son
bruit (les points d'une seule couleur et le grain s'y voient).

**Heart, Express** — Siril, OLD, S1 (retenue), V2 et V2-auto (la Classique, que la règle conseille ici) : la
dominante orangée de la Classique se voit sur les images finales.

![heart express](bench/crossed_heart_express.jpg)

**Iris, Express** — Siril, OLD, S1, V2 : le cas où Siril garde le plus d'avance.

![iris express](bench/crossed_iris_express.jpg)

**Rosette, Base** — Siril, OLD (échelle 1,5), S1 et V2 (échelle 1) : le ciel magenta et les couleurs de l'échelle 1.

![rosette base](bench/crossed_rosette_base.jpg)

### Les tableaux complets (Express)

Une table par mesure : chaque case donne la valeur, puis entre parenthèses l'écart à OLD et l'écart à Siril ;
▲ / ▼ : mieux / moins bien qu'OLD au-delà de la variation d'un passage à l'autre.

<details><summary>Ouvrir les tableaux (Express : 19 configurations, 14 mesures)</summary>

<!-- crossed:express -->
**Profondeur : étoile détectable à 5σ (magnitude G ; plus haut = plus profond)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 16,36 | 17,45 | 18,24 | 17,74 | 16,04 | 17,34 |
| OLD | 16,22 (– / −0,14) | 17,29 (– / −0,16) | 18,22 (– / −0,01) | 17,89 (– / +0,15) | 15,92 (– / −0,12) | 17,06 (– / −0,27) |
| OLD-bis | 16,22 (+0,00 / −0,13) | 17,32 (+0,03 / −0,12) | 18,19 (−0,03 / −0,05) | 17,89 (−0,00 / +0,15) | 15,93 (+0,01 / −0,11) | 17,05 (−0,01 / −0,29) |
| S1 | 16,25 (+0,03 / −0,11) | 17,29 (+0,00 / −0,15) | 18,24 (+0,01 / −0,00) | 17,90 (+0,01 / +0,16) | 15,94 (+0,03 / −0,10) | 17,07 (+0,01 / −0,27) |
| S2 | 16,19 (−0,03 / −0,17) | 17,27 (−0,02 / −0,18) | 18,15 (−0,07 / −0,08) ▼ | 17,89 (−0,00 / +0,15) | 15,93 (+0,01 / −0,11) | 17,04 (−0,02 / −0,29) |
| S3 | 16,21 (−0,01 / −0,15) | 17,30 (+0,02 / −0,14) | 18,20 (−0,02 / −0,04) | 17,88 (−0,01 / +0,14) | 15,92 (+0,00 / −0,12) | 17,02 (−0,04 / −0,32) ▼ |
| S4 | 16,24 (+0,02 / −0,12) | 17,28 (−0,01 / −0,17) | 18,01 (−0,22 / −0,23) ▼ | 17,80 (−0,09 / +0,06) ▼ | 15,80 (−0,11 / −0,24) ▼ | 17,10 (+0,04 / −0,23) ▲ |
| S4b | 16,24 (+0,02 / −0,12) | 17,28 (−0,01 / −0,17) | 18,01 (−0,22 / −0,23) ▼ | 17,80 (−0,09 / +0,06) ▼ | 15,80 (−0,11 / −0,24) ▼ | 17,10 (+0,04 / −0,23) ▲ |
| S5 | 16,21 (−0,01 / −0,15) | 17,29 (+0,00 / −0,15) | 18,19 (−0,03 / −0,05) | 17,89 (−0,00 / +0,15) | 15,92 (+0,00 / −0,12) | 17,03 (−0,03 / −0,31) |
| S6 | 16,23 (+0,01 / −0,13) | 17,31 (+0,03 / −0,13) | 18,21 (−0,01 / −0,03) | 17,89 (−0,00 / +0,15) | 15,92 (+0,01 / −0,12) | 17,02 (−0,04 / −0,31) |
| S7 | 16,36 (+0,13 / −0,00) ▲ | 17,46 (+0,17 / +0,01) ▲ | 18,19 (−0,03 / −0,04) | 17,76 (−0,13 / +0,02) ▼ | 16,03 (+0,11 / −0,01) ▲ | 17,20 (+0,14 / −0,13) ▲ |
| V2 | 16,22 (+0,00 / −0,13) | 17,15 (−0,13 / −0,29) ▼ | 18,17 (−0,05 / −0,06) ▼ | 17,91 (+0,02 / +0,17) | 15,93 (+0,01 / −0,11) | 17,07 (+0,01 / −0,27) |
| V2-auto | 16,38 (+0,16 / +0,02) ▲ | 17,15 (−0,13 / −0,29) ▼ | 18,17 (−0,05 / −0,06) ▼ | 17,91 (+0,02 / +0,17) | 15,93 (+0,01 / −0,11) | 17,07 (+0,01 / −0,27) |
| V2-C | 16,38 (+0,16 / +0,02) ▲ | 17,42 (+0,13 / −0,03) ▲ | 18,26 (+0,04 / +0,03) ▲ | 17,75 (−0,15 / +0,00) ▼ | 16,00 (+0,08 / −0,04) ▲ | 17,19 (+0,13 / −0,14) ▲ |
| V2-S1 | 16,23 (+0,01 / −0,13) | 17,27 (−0,01 / −0,17) | 18,02 (−0,20 / −0,21) ▼ | 17,86 (−0,04 / +0,11) | 15,78 (−0,13 / −0,26) ▼ | 17,07 (+0,01 / −0,27) |
| V2-S2 | 16,24 (+0,02 / −0,12) | 17,17 (−0,12 / −0,28) ▼ | 18,24 (+0,02 / +0,00) | 17,93 (+0,03 / +0,18) | 15,93 (+0,01 / −0,11) | 17,11 (+0,04 / −0,23) ▲ |
| V2-S3 | 16,22 (+0,00 / −0,13) | 17,29 (+0,00 / −0,15) | 18,21 (−0,01 / −0,03) | 17,88 (−0,01 / +0,14) | 15,90 (−0,01 / −0,14) | 17,06 (−0,00 / −0,28) |
| V2-S4 | 16,22 (+0,00 / −0,13) | 17,21 (−0,08 / −0,24) ▼ | 18,17 (−0,05 / −0,06) ▼ | 17,91 (+0,02 / +0,17) | 15,97 (+0,05 / −0,07) ▲ | 17,05 (−0,01 / −0,29) |
| V2-S5 | 16,27 (+0,05 / −0,09) ▲ | 17,15 (−0,14 / −0,30) ▼ | 18,22 (+0,00 / −0,01) | 17,87 (−0,03 / +0,12) | 15,93 (+0,01 / −0,11) | 17,10 (+0,04 / −0,23) ▲ |
| V2-S6 | 16,25 (+0,03 / −0,11) | 17,17 (−0,12 / −0,28) ▼ | 18,20 (−0,02 / −0,04) | 17,94 (+0,05 / +0,20) ▲ | 15,93 (+0,01 / −0,11) | 17,07 (+0,01 / −0,27) |

**Bruit du fond, petite échelle (mag/arcsec² ; plus haut = moins de bruit)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 24,60 | 23,86 | 24,64 | 24,08 | 22,62 | 23,63 |
| OLD | 24,48 (– / −0,13) | 23,69 (– / −0,17) | 24,54 (– / −0,11) | 24,05 (– / −0,04) | 22,51 (– / −0,11) | 23,47 (– / −0,15) |
| OLD-bis | 24,48 (−0,00 / −0,13) | 23,68 (−0,00 / −0,17) | 24,54 (+0,00 / −0,11) | 24,05 (+0,00 / −0,04) | 22,51 (−0,00 / −0,11) | 23,47 (+0,00 / −0,15) |
| S1 | 24,50 (+0,03 / −0,10) | 23,73 (+0,04 / −0,13) ▲ | 24,57 (+0,04 / −0,07) | 24,07 (+0,02 / −0,02) | 22,54 (+0,03 / −0,08) | 23,51 (+0,04 / −0,11) ▲ |
| S2 | 24,47 (−0,01 / −0,14) | 23,67 (−0,02 / −0,19) | 24,50 (−0,03 / −0,14) | 24,04 (−0,00 / −0,04) | 22,50 (−0,01 / −0,12) | 23,47 (−0,01 / −0,16) |
| S3 | 24,48 (+0,00 / −0,12) | 23,69 (+0,01 / −0,16) | 24,54 (+0,01 / −0,10) | 24,05 (−0,00 / −0,04) | 22,52 (+0,01 / −0,10) | 23,48 (+0,01 / −0,15) |
| S4 | 24,48 (−0,00 / −0,13) | 23,66 (−0,02 / −0,20) | 24,38 (−0,15 / −0,26) ▼ | 23,97 (−0,07 / −0,11) ▼ | 22,39 (−0,12 / −0,23) ▼ | 23,52 (+0,05 / −0,10) ▲ |
| S4b | 24,48 (−0,00 / −0,13) | 23,66 (−0,02 / −0,20) | 24,38 (−0,15 / −0,26) ▼ | 23,97 (−0,07 / −0,11) ▼ | 22,39 (−0,12 / −0,23) ▼ | 23,52 (+0,05 / −0,10) ▲ |
| S5 | 24,48 (−0,00 / −0,13) | 23,68 (−0,00 / −0,17) | 24,54 (+0,00 / −0,11) | 24,05 (+0,00 / −0,04) | 22,51 (−0,00 / −0,11) | 23,47 (+0,00 / −0,15) |
| S6 | 24,47 (−0,00 / −0,13) | 23,69 (+0,00 / −0,17) | 24,54 (−0,00 / −0,11) | 24,05 (+0,01 / −0,03) | 22,51 (+0,00 / −0,11) | 23,46 (−0,01 / −0,16) |
| S7 | 24,61 (+0,13 / +0,00) ▲ | 23,87 (+0,18 / +0,01) ▲ | 24,64 (+0,11 / +0,00) ▲ | 24,09 (+0,04 / +0,01) ▲ | 22,64 (+0,13 / +0,02) ▲ | 23,64 (+0,17 / +0,01) ▲ |
| V2 | 24,50 (+0,02 / −0,10) | 23,69 (+0,00 / −0,17) | 24,56 (+0,03 / −0,08) | 24,05 (+0,01 / −0,03) | 22,51 (+0,00 / −0,11) | 23,54 (+0,07 / −0,08) ▲ |
| V2-auto | 24,61 (+0,13 / +0,01) ▲ | 23,69 (+0,00 / −0,17) | 24,56 (+0,03 / −0,08) | 24,05 (+0,01 / −0,03) | 22,51 (+0,00 / −0,11) | 23,54 (+0,07 / −0,08) ▲ |
| V2-C | 24,61 (+0,13 / +0,01) ▲ | 23,84 (+0,15 / −0,02) ▲ | 24,65 (+0,11 / +0,01) ▲ | 24,08 (+0,04 / −0,00) | 22,61 (+0,10 / −0,01) ▲ | 23,68 (+0,20 / +0,05) ▲ |
| V2-S1 | 24,47 (−0,00 / −0,13) | 23,65 (−0,03 / −0,21) | 24,37 (−0,17 / −0,27) ▼ | 23,98 (−0,07 / −0,11) ▼ | 22,40 (−0,11 / −0,22) ▼ | 23,53 (+0,05 / −0,10) ▲ |
| V2-S2 | 24,51 (+0,03 / −0,09) | 23,70 (+0,02 / −0,16) | 24,58 (+0,04 / −0,06) ▲ | 24,06 (+0,01 / −0,02) | 22,52 (+0,01 / −0,10) | 23,55 (+0,08 / −0,07) ▲ |
| V2-S3 | 24,49 (+0,02 / −0,11) | 23,68 (−0,01 / −0,18) | 24,56 (+0,02 / −0,08) | 24,05 (+0,01 / −0,03) | 22,51 (−0,00 / −0,11) | 23,55 (+0,07 / −0,08) ▲ |
| V2-S4 | 24,50 (+0,02 / −0,10) | 23,72 (+0,04 / −0,14) | 24,56 (+0,03 / −0,08) | 24,06 (+0,02 / −0,02) | 22,54 (+0,03 / −0,07) | 23,51 (+0,04 / −0,12) |
| V2-S5 | 24,50 (+0,02 / −0,10) | 23,68 (−0,00 / −0,18) | 24,56 (+0,03 / −0,08) | 24,06 (+0,01 / −0,03) | 22,51 (+0,00 / −0,11) | 23,54 (+0,07 / −0,08) ▲ |
| V2-S6 | 24,50 (+0,02 / −0,10) | 23,68 (−0,00 / −0,17) | 24,56 (+0,02 / −0,08) | 24,05 (+0,01 / −0,03) | 22,52 (+0,01 / −0,10) | 23,54 (+0,07 / −0,08) ▲ |

**Bruit du fond, échelle 4 fois plus grande (mag/arcsec²)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 25,29 | 25,22 | 25,87 | 24,38 | 23,78 | 24,81 |
| OLD | 25,24 (– / −0,05) | 25,10 (– / −0,12) | 25,88 (– / +0,01) | 24,35 (– / −0,03) | 23,68 (– / −0,10) | 24,73 (– / −0,08) |
| OLD-bis | 25,25 (+0,00 / −0,04) | 25,10 (−0,00 / −0,12) | 25,88 (+0,00 / +0,01) | 24,35 (+0,00 / −0,03) | 23,66 (−0,03 / −0,13) | 24,74 (+0,00 / −0,07) |
| S1 | 25,25 (+0,00 / −0,04) | 25,12 (+0,02 / −0,11) | 25,89 (+0,02 / +0,02) | 24,36 (+0,01 / −0,02) | 23,71 (+0,02 / −0,08) | 24,75 (+0,01 / −0,06) |
| S2 | 25,23 (−0,01 / −0,05) | 25,08 (−0,02 / −0,14) | 25,85 (−0,03 / −0,02) | 24,35 (−0,00 / −0,03) | 23,66 (−0,02 / −0,12) | 24,73 (−0,01 / −0,08) |
| S3 | 25,24 (+0,00 / −0,04) | 25,10 (−0,00 / −0,13) | 25,88 (+0,01 / +0,02) | 24,36 (+0,01 / −0,02) | 23,64 (−0,05 / −0,15) ▼ | 24,74 (+0,01 / −0,07) |
| S4 | 25,24 (−0,00 / −0,05) | 25,08 (−0,02 / −0,15) | 25,69 (−0,18 / −0,18) ▼ | 24,35 (+0,00 / −0,03) | 23,55 (−0,13 / −0,23) ▼ | 24,77 (+0,04 / −0,04) |
| S4b | 25,24 (−0,00 / −0,05) | 25,08 (−0,02 / −0,15) | 25,69 (−0,18 / −0,18) ▼ | 24,35 (+0,00 / −0,03) | 23,55 (−0,13 / −0,23) ▼ | 24,77 (+0,04 / −0,04) |
| S5 | 25,24 (+0,00 / −0,04) | 25,11 (+0,01 / −0,12) | 25,88 (+0,01 / +0,01) | 24,36 (+0,01 / −0,02) | 23,67 (−0,02 / −0,12) | 24,73 (−0,00 / −0,08) |
| S6 | 25,24 (+0,00 / −0,05) | 25,09 (−0,01 / −0,13) | 25,88 (+0,00 / +0,01) | 24,36 (+0,01 / −0,02) | 23,66 (−0,02 / −0,13) | 24,71 (−0,03 / −0,10) |
| S7 | 25,29 (+0,05 / +0,01) ▲ | 25,22 (+0,12 / +0,00) ▲ | 25,86 (−0,02 / −0,01) | 24,38 (+0,03 / −0,00) | 23,76 (+0,08 / −0,02) ▲ | 24,82 (+0,09 / +0,01) ▲ |
| V2 | 25,24 (−0,00 / −0,05) | 25,08 (−0,02 / −0,14) | 25,89 (+0,01 / +0,02) | 24,35 (+0,00 / −0,03) | 23,67 (−0,02 / −0,12) | 24,76 (+0,03 / −0,05) |
| V2-auto | 25,30 (+0,06 / +0,01) ▲ | 25,08 (−0,02 / −0,14) | 25,89 (+0,01 / +0,02) | 24,35 (+0,00 / −0,03) | 23,67 (−0,02 / −0,12) | 24,76 (+0,03 / −0,05) |
| V2-C | 25,30 (+0,06 / +0,01) ▲ | 25,18 (+0,08 / −0,04) ▲ | 25,91 (+0,03 / +0,04) | 24,38 (+0,02 / −0,00) | 23,76 (+0,07 / −0,03) ▲ | 24,85 (+0,12 / +0,04) ▲ |
| V2-S1 | 25,24 (+0,00 / −0,05) | 25,08 (−0,02 / −0,14) | 25,68 (−0,20 / −0,19) ▼ | 24,36 (+0,01 / −0,02) | 23,59 (−0,10 / −0,20) ▼ | 24,78 (+0,05 / −0,03) ▲ |
| V2-S2 | 25,25 (+0,00 / −0,04) | 25,09 (−0,01 / −0,14) | 25,90 (+0,03 / +0,03) | 24,36 (+0,01 / −0,02) | 23,67 (−0,02 / −0,12) | 24,78 (+0,05 / −0,03) ▲ |
| V2-S3 | 25,24 (−0,00 / −0,05) | 25,07 (−0,03 / −0,15) | 25,88 (+0,01 / +0,01) | 24,35 (+0,00 / −0,03) | 23,68 (−0,00 / −0,10) | 24,77 (+0,04 / −0,03) ▲ |
| V2-S4 | 25,24 (−0,00 / −0,05) | 25,12 (+0,01 / −0,11) | 25,89 (+0,01 / +0,02) | 24,35 (+0,00 / −0,02) | 23,73 (+0,04 / −0,06) ▲ | 24,74 (+0,00 / −0,07) |
| V2-S5 | 25,24 (−0,00 / −0,05) | 25,08 (−0,02 / −0,14) | 25,88 (+0,01 / +0,01) | 24,36 (+0,01 / −0,02) | 23,66 (−0,02 / −0,12) | 24,77 (+0,03 / −0,04) |
| V2-S6 | 25,24 (−0,00 / −0,05) | 25,09 (−0,02 / −0,14) | 25,89 (+0,01 / +0,02) | 24,35 (−0,00 / −0,03) | 23,68 (−0,01 / −0,11) | 24,77 (+0,04 / −0,04) |

**Signal/bruit sur la cible**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 5,0 | 3,0 | 4,7 | 48,8 | 22,0 | 4,9 |
| OLD | 4,4 (– / −12 %) | 2,3 (– / −21 %) | 4,1 (– / −12 %) | 45,6 (– / −7 %) | 22,8 (– / +3 %) | 4,2 (– / −15 %) |
| OLD-bis | 4,5 (+0 % / −12 %) | 2,3 (−0 % / −21 %) | 4,1 (+0 % / −12 %) | 45,5 (−0 % / −7 %) | 22,8 (+0 % / +4 %) | 4,2 (+0 % / −15 %) |
| S1 | 4,7 (+5 % / −8 %) ▲ | 2,4 (+2 % / −20 %) | 4,2 (+4 % / −9 %) ▲ | 46,4 (+2 % / −5 %) | 24,1 (+6 % / +9 %) ▲ | 4,3 (+3 % / −12 %) ▲ |
| S2 | 4,5 (+1 % / −12 %) | 2,3 (−1 % / −22 %) | 4,1 (−0 % / −13 %) | 45,6 (−0 % / −7 %) | 22,8 (−0 % / +3 %) | 4,2 (+0 % / −15 %) |
| S3 | 4,5 (+2 % / −10 %) | 2,3 (−0 % / −21 %) | 4,1 (+0 % / −12 %) | 45,7 (+0 % / −6 %) | 23,7 (+4 % / +8 %) ▲ | 4,2 (+1 % / −14 %) |
| S4 | 4,4 (+0 % / −12 %) | 2,3 (−3 % / −24 %) ▼ | 3,5 (−14 % / −25 %) ▼ | 42,3 (−7 % / −13 %) ▼ | 20,5 (−10 % / −7 %) ▼ | 4,5 (+8 % / −9 %) ▲ |
| S4b | 4,4 (+0 % / −12 %) | 2,3 (−3 % / −24 %) ▼ | 3,5 (−14 % / −25 %) ▼ | 42,3 (−7 % / −13 %) ▼ | 20,5 (−10 % / −7 %) ▼ | 4,5 (+8 % / −9 %) ▲ |
| S5 | 4,5 (+2 % / −11 %) | 2,4 (+1 % / −20 %) | 3,7 (−11 % / −22 %) ▼ | 45,5 (−0 % / −7 %) | 22,8 (−0 % / +3 %) | 4,2 (+0 % / −15 %) |
| S6 | 4,4 (+0 % / −12 %) | 2,3 (−1 % / −22 %) | 4,1 (−0 % / −13 %) | 45,5 (−0 % / −7 %) | 22,5 (−1 % / +2 %) | 4,2 (+1 % / −14 %) |
| S7 | 4,8 (+9 % / −4 %) ▲ | 2,5 (+8 % / −14 %) ▲ | 4,5 (+11 % / −3 %) ▲ | 49,9 (+10 % / +2 %) ▲ | 24,7 (+9 % / +12 %) ▲ | 4,6 (+10 % / −6 %) ▲ |
| V2 | 4,7 (+6 % / −7 %) ▲ | 2,4 (+1 % / −20 %) | 3,8 (−8 % / −19 %) ▼ | 45,9 (+1 % / −6 %) | 22,9 (+0 % / +4 %) | 4,6 (+9 % / −7 %) ▲ |
| V2-auto | 4,9 (+10 % / −3 %) ▲ | 2,4 (+1 % / −20 %) | 3,8 (−8 % / −19 %) ▼ | 45,9 (+1 % / −6 %) | 22,9 (+0 % / +4 %) | 4,6 (+9 % / −7 %) ▲ |
| V2-C | 4,9 (+10 % / −3 %) ▲ | 2,5 (+8 % / −15 %) ▲ | 4,1 (−1 % / −13 %) | 49,4 (+8 % / +1 %) ▲ | 24,0 (+5 % / +9 %) ▲ | 4,9 (+17 % / −0 %) ▲ |
| V2-S1 | 4,6 (+3 % / −9 %) ▲ | 2,3 (−1 % / −22 %) | 3,1 (−24 % / −33 %) ▼ | 42,6 (−6 % / −13 %) ▼ | 20,9 (−9 % / −5 %) ▼ | 4,5 (+8 % / −8 %) ▲ |
| V2-S2 | 4,7 (+5 % / −7 %) ▲ | 2,4 (+1 % / −20 %) | 3,8 (−7 % / −19 %) ▼ | 45,9 (+1 % / −6 %) | 23,3 (+2 % / +6 %) | 4,6 (+9 % / −8 %) ▲ |
| V2-S3 | 4,7 (+6 % / −7 %) ▲ | 2,4 (+1 % / −20 %) | 3,8 (−8 % / −19 %) ▼ | 45,8 (+1 % / −6 %) | 22,8 (+0 % / +4 %) | 4,6 (+10 % / −6 %) ▲ |
| V2-S4 | 4,7 (+6 % / −7 %) ▲ | 2,5 (+5 % / −17 %) ▲ | 3,8 (−8 % / −19 %) ▼ | 46,8 (+3 % / −4 %) | 24,4 (+7 % / +11 %) ▲ | 4,4 (+6 % / −10 %) ▲ |
| V2-S5 | 4,7 (+5 % / −7 %) ▲ | 2,4 (+1 % / −20 %) | 4,2 (+3 % / −9 %) ▲ | 45,8 (+1 % / −6 %) | 23,5 (+3 % / +7 %) | 4,6 (+10 % / −6 %) ▲ |
| V2-S6 | 4,7 (+6 % / −7 %) ▲ | 2,4 (+2 % / −19 %) | 3,8 (−7 % / −19 %) ▼ | 45,7 (+0 % / −6 %) | 22,7 (−1 % / +3 %) | 4,6 (+9 % / −7 %) ▲ |

**Étoiles faibles trouvées à 5σ (les mêmes étoiles Gaia pour toutes)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 22159 | 4795 | 1223 | 26990 | 570 | 5491 |
| OLD | 21903 (– / −1 %) | 4606 (– / −4 %) | 1208 (– / −1 %) | 27601 (– / +2 %) | 548 (– / −4 %) | 5255 (– / −4 %) |
| OLD-bis | 21888 (−0 % / −1 %) | 4675 (+1 % / −3 %) | 1204 (−0 % / −2 %) | 27613 (+0 % / +2 %) | 548 (+0 % / −4 %) | 5230 (−0 % / −5 %) |
| S1 | 21892 (−0 % / −1 %) | 4597 (−0 % / −4 %) | 1219 (+1 % / −0 %) | 27709 (+0 % / +3 %) | 552 (+1 % / −3 %) | 5218 (−1 % / −5 %) |
| S2 | 21816 (−0 % / −2 %) | 4573 (−1 % / −5 %) | 1204 (−0 % / −2 %) | 27587 (−0 % / +2 %) | 552 (+1 % / −3 %) | 5221 (−1 % / −5 %) |
| S3 | 21836 (−0 % / −1 %) | 4580 (−1 % / −4 %) | 1215 (+1 % / −1 %) | 27521 (−0 % / +2 %) | 548 (+0 % / −4 %) | 5166 (−2 % / −6 %) |
| S4 | 21901 (−0 % / −1 %) | 4585 (−0 % / −4 %) | 1145 (−5 % / −6 %) ▼ | 27537 (−0 % / +2 %) | 521 (−5 % / −9 %) ▼ | 5337 (+2 % / −3 %) |
| S4b | 21901 (−0 % / −1 %) | 4585 (−0 % / −4 %) | 1145 (−5 % / −6 %) ▼ | 27537 (−0 % / +2 %) | 521 (−5 % / −9 %) ▼ | 5337 (+2 % / −3 %) |
| S5 | 21909 (+0 % / −1 %) | 4618 (+0 % / −4 %) | 1199 (−1 % / −2 %) | 27658 (+0 % / +2 %) | 553 (+1 % / −3 %) | 5245 (−0 % / −4 %) |
| S6 | 21933 (+0 % / −1 %) | 4673 (+1 % / −3 %) | 1218 (+1 % / −0 %) | 27635 (+0 % / +2 %) | 556 (+1 % / −2 %) | 5187 (−1 % / −6 %) |
| S7 | 22166 (+1 % / +0 %) | 4809 (+4 % / +0 %) ▲ | 1211 (+0 % / −1 %) | 26975 (−2 % / −0 %) | 571 (+4 % / +0 %) ▲ | 5407 (+3 % / −2 %) |
| V2 | 21820 (−0 % / −2 %) | 4461 (−3 % / −7 %) ▼ | 1211 (+0 % / −1 %) | 27732 (+0 % / +3 %) | 555 (+1 % / −3 %) | 5217 (−1 % / −5 %) |
| V2-auto | 22221 (+1 % / +0 %) | 4461 (−3 % / −7 %) ▼ | 1211 (+0 % / −1 %) | 27732 (+0 % / +3 %) | 555 (+1 % / −3 %) | 5217 (−1 % / −5 %) |
| V2-C | 22221 (+1 % / +0 %) | 4764 (+3 % / −1 %) ▲ | 1225 (+1 % / +0 %) | 26802 (−3 % / −1 %) | 557 (+2 % / −2 %) | 5412 (+3 % / −1 %) |
| V2-S1 | 21925 (+0 % / −1 %) | 4552 (−1 % / −5 %) | 1146 (−5 % / −6 %) ▼ | 27588 (−0 % / +2 %) | 508 (−7 % / −11 %) ▼ | 5242 (−0 % / −5 %) |
| V2-S2 | 21922 (+0 % / −1 %) | 4499 (−2 % / −6 %) | 1220 (+1 % / −0 %) | 27803 (+1 % / +3 %) | 554 (+1 % / −3 %) | 5304 (+1 % / −3 %) |
| V2-S3 | 21911 (+0 % / −1 %) | 4652 (+1 % / −3 %) | 1225 (+1 % / +0 %) | 27638 (+0 % / +2 %) | 538 (−2 % / −6 %) | 5254 (−0 % / −4 %) |
| V2-S4 | 21820 (−0 % / −2 %) | 4554 (−1 % / −5 %) | 1211 (+0 % / −1 %) | 27685 (+0 % / +3 %) | 552 (+1 % / −3 %) | 5218 (−1 % / −5 %) |
| V2-S5 | 22008 (+0 % / −1 %) | 4472 (−3 % / −7 %) | 1217 (+1 % / −0 %) | 27659 (+0 % / +2 %) | 550 (+0 % / −4 %) | 5330 (+1 % / −3 %) |
| V2-S6 | 21895 (−0 % / −1 %) | 4553 (−1 % / −5 %) | 1211 (+0 % / −1 %) | 27802 (+1 % / +3 %) | 539 (−2 % / −5 %) | 5271 (+0 % / −4 %) |

**Finesse des étoiles : FWHM au centre (″ ; plus bas = plus fines)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 13,10 | 6,11 | 6,89 | 5,62 | 5,85 | 5,80 |
| OLD | 12,07 (– / −8 %) | 6,14 (– / +0 %) | 6,93 (– / +1 %) | 5,32 (– / −5 %) | 5,95 (– / +2 %) | 5,98 (– / +3 %) |
| OLD-bis | 12,07 (+0 % / −8 %) | 6,14 (+0 % / +0 %) | 6,94 (+0 % / +1 %) | 5,32 (+0 % / −5 %) | 5,95 (+0 % / +2 %) | 5,98 (−0 % / +3 %) |
| S1 | 12,56 (+4 % / −4 %) ▼ | 6,24 (+2 % / +2 %) | 6,99 (+1 % / +1 %) | 5,44 (+2 % / −3 %) | 6,12 (+3 % / +5 %) | 6,05 (+1 % / +4 %) |
| S2 | 12,11 (+0 % / −8 %) | 6,16 (+0 % / +1 %) | 6,93 (+0 % / +1 %) | 5,34 (+0 % / −5 %) | 5,98 (+1 % / +2 %) | 5,97 (−0 % / +3 %) |
| S3 | 12,08 (+0 % / −8 %) | 6,22 (+1 % / +2 %) | 7,01 (+1 % / +2 %) | 5,34 (+1 % / −5 %) | 6,02 (+1 % / +3 %) | 6,00 (+0 % / +3 %) |
| S4 | 12,07 (+0 % / −8 %) | 6,09 (−1 % / −0 %) | 6,76 (−2 % / −2 %) | 5,18 (−2 % / −8 %) | 5,84 (−2 % / −0 %) | 5,93 (−1 % / +2 %) |
| S4b | 12,07 (+0 % / −8 %) | 6,09 (−1 % / −0 %) | 6,76 (−2 % / −2 %) | 5,18 (−2 % / −8 %) | 5,84 (−2 % / −0 %) | 5,93 (−1 % / +2 %) |
| S5 | 12,12 (+0 % / −7 %) | 6,14 (−0 % / +0 %) | 6,93 (+0 % / +1 %) | 5,33 (+0 % / −5 %) | 5,96 (+0 % / +2 %) | 5,98 (+0 % / +3 %) |
| S6 | 12,08 (+0 % / −8 %) | 6,14 (−0 % / +0 %) | 6,94 (+0 % / +1 %) | 5,33 (+0 % / −5 %) | 5,95 (+0 % / +2 %) | 5,97 (−0 % / +3 %) |
| S7 | 13,02 (+8 % / −1 %) ▼ | 6,08 (−1 % / −1 %) | 6,85 (−1 % / −1 %) | 5,58 (+5 % / −1 %) ▼ | 5,84 (−2 % / −0 %) | 5,91 (−1 % / +2 %) |
| V2 | 12,69 (+5 % / −3 %) ▼ | 6,30 (+3 % / +3 %) | 7,05 (+2 % / +2 %) | 5,46 (+3 % / −3 %) | 6,10 (+3 % / +4 %) | 6,01 (+1 % / +4 %) |
| V2-auto | 13,08 (+8 % / −0 %) ▼ | 6,30 (+3 % / +3 %) | 7,05 (+2 % / +2 %) | 5,46 (+3 % / −3 %) | 6,10 (+3 % / +4 %) | 6,01 (+1 % / +4 %) |
| V2-C | 13,08 (+8 % / −0 %) ▼ | 6,12 (−0 % / +0 %) | 6,92 (−0 % / +0 %) | 5,56 (+5 % / −1 %) ▼ | 5,84 (−2 % / −0 %) | 5,86 (−2 % / +1 %) |
| V2-S1 | 12,22 (+1 % / −7 %) | 6,15 (+0 % / +1 %) | 6,79 (−2 % / −1 %) | 5,23 (−2 % / −7 %) | 5,91 (−1 % / +1 %) | 5,95 (−0 % / +3 %) |
| V2-S2 | 12,64 (+5 % / −3 %) ▼ | 6,27 (+2 % / +3 %) | 7,06 (+2 % / +2 %) | 5,43 (+2 % / −3 %) | 6,10 (+3 % / +4 %) | 6,01 (+1 % / +4 %) |
| V2-S3 | 12,69 (+5 % / −3 %) ▼ | 6,23 (+1 % / +2 %) | 6,98 (+1 % / +1 %) | 5,44 (+2 % / −3 %) | 6,07 (+2 % / +4 %) | 5,99 (+0 % / +3 %) |
| V2-S4 | 12,69 (+5 % / −3 %) ▼ | 6,32 (+3 % / +3 %) | 7,05 (+2 % / +2 %) | 5,50 (+3 % / −2 %) ▼ | 6,18 (+4 % / +6 %) ▼ | 6,08 (+2 % / +5 %) |
| V2-S5 | 12,63 (+5 % / −4 %) ▼ | 6,30 (+3 % / +3 %) | 7,06 (+2 % / +2 %) | 5,45 (+2 % / −3 %) | 6,11 (+3 % / +4 %) | 6,01 (+1 % / +4 %) |
| V2-S6 | 12,68 (+5 % / −3 %) ▼ | 6,31 (+3 % / +3 %) | 7,05 (+2 % / +2 %) | 5,46 (+3 % / −3 %) | 6,11 (+3 % / +4 %) | 6,01 (+1 % / +4 %) |

**Rondeur des étoiles au centre (1 = rondes)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 0,91 | 0,89 | 0,82 | 0,93 | 0,89 | 0,76 |
| OLD | 0,91 (– / −0,00) | 0,88 (– / −0,01) | 0,83 (– / +0,01) | 0,95 (– / +0,01) | 0,86 (– / −0,03) | 0,76 (– / −0,00) |
| OLD-bis | 0,91 (+0,00 / −0,00) | 0,88 (+0,00 / −0,01) | 0,83 (+0,00 / +0,01) | 0,95 (+0,00 / +0,01) | 0,86 (−0,00 / −0,03) | 0,76 (+0,00 / −0,00) |
| S1 | 0,91 (+0,01 / +0,00) | 0,88 (+0,00 / −0,00) | 0,84 (+0,00 / +0,01) | 0,95 (+0,00 / +0,02) | 0,87 (+0,01 / −0,02) | 0,76 (+0,01 / +0,00) |
| S2 | 0,91 (+0,00 / −0,00) | 0,88 (+0,00 / −0,01) | 0,83 (+0,00 / +0,01) | 0,95 (+0,00 / +0,01) | 0,86 (+0,00 / −0,03) | 0,76 (+0,00 / −0,00) |
| S3 | 0,91 (+0,00 / −0,00) | 0,88 (+0,00 / −0,01) | 0,83 (−0,01 / +0,00) | 0,95 (−0,00 / +0,01) | 0,85 (−0,01 / −0,04) | 0,76 (−0,00 / −0,01) |
| S4 | 0,91 (+0,00 / −0,00) | 0,88 (−0,00 / −0,01) | 0,84 (+0,01 / +0,02) | 0,95 (+0,00 / +0,01) | 0,88 (+0,02 / −0,01) | 0,75 (−0,01 / −0,01) |
| S4b | 0,91 (+0,00 / −0,00) | 0,88 (−0,00 / −0,01) | 0,84 (+0,01 / +0,02) | 0,95 (+0,00 / +0,01) | 0,88 (+0,02 / −0,01) | 0,75 (−0,01 / −0,01) |
| S5 | 0,91 (+0,00 / −0,00) | 0,88 (+0,00 / −0,01) | 0,83 (−0,00 / +0,01) | 0,95 (−0,00 / +0,01) | 0,86 (+0,00 / −0,03) | 0,76 (+0,00 / −0,00) |
| S6 | 0,91 (+0,00 / −0,00) | 0,88 (−0,00 / −0,01) | 0,83 (+0,00 / +0,01) | 0,95 (−0,00 / +0,01) | 0,86 (+0,00 / −0,03) | 0,76 (+0,00 / −0,00) |
| S7 | 0,92 (+0,02 / +0,01) | 0,89 (+0,01 / +0,00) | 0,83 (−0,00 / +0,01) | 0,93 (−0,02 / −0,00) | 0,92 (+0,05 / +0,03) ▲ | 0,77 (+0,01 / +0,01) |
| V2 | 0,91 (+0,01 / +0,00) | 0,89 (+0,01 / +0,00) | 0,83 (−0,01 / +0,00) | 0,95 (−0,00 / +0,01) | 0,87 (+0,01 / −0,01) | 0,77 (+0,01 / +0,01) |
| V2-auto | 0,92 (+0,02 / +0,01) | 0,89 (+0,01 / +0,00) | 0,83 (−0,01 / +0,00) | 0,95 (−0,00 / +0,01) | 0,87 (+0,01 / −0,01) | 0,77 (+0,01 / +0,01) |
| V2-C | 0,92 (+0,02 / +0,01) | 0,89 (+0,00 / −0,00) | 0,82 (−0,01 / −0,00) | 0,93 (−0,02 / −0,01) | 0,92 (+0,06 / +0,03) ▲ | 0,77 (+0,01 / +0,01) |
| V2-S1 | 0,90 (−0,00 / −0,01) | 0,88 (−0,00 / −0,01) | 0,84 (+0,01 / +0,02) | 0,95 (+0,00 / +0,01) | 0,88 (+0,02 / −0,01) | 0,75 (−0,01 / −0,01) |
| V2-S2 | 0,91 (+0,01 / +0,00) | 0,89 (+0,01 / +0,00) | 0,83 (−0,01 / +0,00) | 0,95 (−0,00 / +0,01) | 0,87 (+0,01 / −0,01) | 0,77 (+0,01 / +0,01) |
| V2-S3 | 0,91 (+0,01 / +0,00) | 0,89 (+0,01 / −0,00) | 0,84 (+0,00 / +0,01) | 0,95 (−0,00 / +0,01) | 0,88 (+0,02 / −0,01) ▲ | 0,77 (+0,01 / +0,01) |
| V2-S4 | 0,91 (+0,01 / +0,00) | 0,89 (+0,01 / −0,00) | 0,83 (−0,01 / +0,00) | 0,95 (−0,00 / +0,01) | 0,86 (−0,01 / −0,03) | 0,76 (+0,00 / −0,00) |
| V2-S5 | 0,91 (+0,01 / +0,00) | 0,89 (+0,01 / −0,00) | 0,83 (−0,01 / +0,00) | 0,95 (−0,00 / +0,01) | 0,87 (+0,01 / −0,01) | 0,77 (+0,01 / +0,01) |
| V2-S6 | 0,91 (+0,01 / +0,00) | 0,89 (+0,01 / +0,00) | 0,83 (−0,01 / +0,00) | 0,95 (−0,00 / +0,01) | 0,87 (+0,01 / −0,01) | 0,77 (+0,01 / +0,01) |

**Points d'une seule couleur (pixels chauds restants, par degré²)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 1 | 3 | 4 | 1418 | 0 | 7 |
| OLD | 21 (– / +20) | 25 (– / +22) | 18 (– / +13) | 2323 (– / +905) | 5 (– / +5) | 21 (– / +15) |
| OLD-bis | 21 (+0 / +20) | 26 (+1 / +23) | 18 (−0 / +13) | 2326 (+3 / +908) | 5 (−0 / +5) | 22 (+1 / +15) |
| S1 | 24 (+3 / +23) | 21 (−4 / +18) ▲ | 17 (−0 / +13) | 2472 (+149 / +1054) | 8 (+3 / +8) | 22 (+0 / +15) |
| S2 | 22 (+1 / +21) | 28 (+3 / +25) | 32 (+15 / +28) ▼ | 2433 (+110 / +1014) | 5 (−0 / +5) | 25 (+4 / +19) ▼ |
| S3 | 22 (+1 / +21) | 24 (−1 / +21) | 16 (−2 / +12) | 2351 (+28 / +933) | 8 (+3 / +8) | 22 (+1 / +15) |
| S4 | 21 (+0 / +20) | 27 (+1 / +24) | 16 (−1 / +12) | 2135 (−188 / +717) | 3 (−3 / +3) | 14 (−8 / +7) ▲ |
| S4b | 21 (+0 / +20) | 27 (+1 / +24) | 16 (−1 / +12) | 2132 (−191 / +714) | 3 (−3 / +3) | 14 (−8 / +7) ▲ |
| S5 | 21 (+0 / +20) | 25 (+0 / +22) | 18 (+1 / +14) | 2323 (+0 / +905) | 8 (+3 / +8) | 21 (−1 / +14) |
| S6 | 22 (+1 / +21) | 25 (−0 / +22) | 17 (−1 / +13) | 2446 (+123 / +1027) | 5 (−0 / +5) | 24 (+2 / +17) |
| S7 | 1 (−20 / −0) ▲ | 6 (−19 / +3) ▲ | 5 (−12 / +1) ▲ | 1175 (−1148 / −243) ▲ | 5 (−0 / +5) | 7 (−15 / +0) ▲ |
| V2 | 26 (+5 / +25) ▼ | 22 (−3 / +19) | 19 (+1 / +14) | 2528 (+205 / +1110) | 5 (−0 / +5) | 17 (−4 / +10) ▲ |
| V2-auto | 1 (−20 / −0) ▲ | 22 (−3 / +19) | 19 (+1 / +14) | 2528 (+205 / +1110) | 5 (−0 / +5) | 17 (−4 / +10) ▲ |
| V2-C | 1 (−20 / −0) ▲ | 7 (−19 / +4) ▲ | 5 (−13 / +1) ▲ | 1151 (−1172 / −267) ▲ | 3 (−3 / +3) | 7 (−15 / +0) ▲ |
| V2-S1 | 26 (+5 / +25) ▼ | 30 (+5 / +27) ▼ | 36 (+19 / +32) ▼ | 2298 (−25 / +880) | 3 (−3 / +3) | 23 (+2 / +16) |
| V2-S2 | 26 (+5 / +25) ▼ | 20 (−6 / +17) ▲ | 17 (−0 / +13) | 2547 (+225 / +1129) | 5 (−0 / +5) | 18 (−4 / +11) ▲ |
| V2-S3 | 24 (+3 / +23) | 23 (−2 / +20) | 20 (+2 / +15) | 2507 (+185 / +1089) | 3 (−3 / +3) | 14 (−7 / +7) ▲ |
| V2-S4 | 26 (+5 / +25) ▼ | 22 (−4 / +19) | 19 (+1 / +14) | 2545 (+223 / +1127) | 3 (−3 / +3) | 22 (+1 / +16) |
| V2-S5 | 26 (+4 / +24) ▼ | 20 (−5 / +17) ▲ | 18 (+0 / +14) | 2546 (+223 / +1128) | 5 (−0 / +5) | 19 (−2 / +12) |
| V2-S6 | 24 (+3 / +23) ▼ | 20 (−6 / +17) ▲ | 18 (+0 / +14) | 2439 (+116 / +1021) | 5 (−0 / +5) | 18 (−3 / +12) |

**Taux de rejet (% des échantillons ; repère de Siril : 0,1 à 0,5 %)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 0,941 | 0,892 | 0,804 | 1,222 | 0,919 | 0,910 |
| OLD | 0,014 (– / −0,927) | 0,044 (– / −0,848) | 0,024 (– / −0,780) | 0,014 (– / −1,208) | 0,001 (– / −0,918) | 0,039 (– / −0,871) |
| OLD-bis | 0,014 (+0,000 / −0,927) | 0,044 (+0,000 / −0,848) | 0,024 (+0,000 / −0,780) | 0,014 (+0,000 / −1,208) | 0,001 (+0,000 / −0,918) | 0,039 (+0,000 / −0,871) |
| S1 | 0,014 (+0,000 / −0,927) | 0,045 (+0,001 / −0,847) | 0,025 (+0,001 / −0,779) | 0,014 (+0,000 / −1,208) | 0,001 (+0,000 / −0,918) | 0,038 (−0,001 / −0,872) |
| S2 | 0,025 (+0,011 / −0,916) | 0,218 (+0,174 / −0,674) | 0,041 (+0,017 / −0,763) | 0,008 (−0,006 / −1,214) | 0,004 (+0,003 / −0,915) | 0,192 (+0,153 / −0,718) |
| S3 | 0,014 (+0,000 / −0,927) | 0,044 (+0,000 / −0,848) | 0,024 (+0,000 / −0,780) | 0,014 (+0,000 / −1,208) | 0,001 (+0,000 / −0,918) | 0,039 (+0,000 / −0,871) |
| S4 | 0,014 (+0,000 / −0,927) | 0,044 (+0,000 / −0,848) | 0,021 (−0,003 / −0,783) | 0,016 (+0,002 / −1,206) | 0,001 (+0,000 / −0,918) | 0,040 (+0,001 / −0,870) |
| S4b | 0,014 (+0,000 / −0,927) | 0,044 (+0,000 / −0,848) | 0,021 (−0,003 / −0,783) | 0,016 (+0,002 / −1,206) | 0,001 (+0,000 / −0,918) | 0,040 (+0,001 / −0,870) |
| S5 | 0,014 (+0,000 / −0,927) | 0,044 (+0,000 / −0,848) | 0,024 (+0,000 / −0,780) | 0,014 (+0,000 / −1,208) | 0,001 (+0,000 / −0,918) | 0,039 (+0,000 / −0,871) |
| S6 | 0,014 (+0,000 / −0,927) | 0,044 (+0,000 / −0,848) | 0,024 (+0,000 / −0,780) | 0,014 (+0,000 / −1,208) | 0,001 (+0,000 / −0,918) | 0,039 (+0,000 / −0,871) |
| S7 | 1,097 (+1,083 / +0,156) | 1,347 (+1,303 / +0,455) | 0,852 (+0,828 / +0,048) | 1,215 (+1,201 / −0,007) | 0,959 (+0,958 / +0,040) | 0,545 (+0,506 / −0,365) |
| V2 | 0,020 (+0,006 / −0,921) | 0,145 (+0,101 / −0,747) | 0,046 (+0,022 / −0,758) | 0,013 (−0,001 / −1,209) | 0,004 (+0,003 / −0,915) | 0,173 (+0,134 / −0,737) |
| V2-auto | 1,124 (+1,110 / +0,183) | 0,145 (+0,101 / −0,747) | 0,046 (+0,022 / −0,758) | 0,013 (−0,001 / −1,209) | 0,004 (+0,003 / −0,915) | 0,173 (+0,134 / −0,737) |
| V2-C | 1,124 (+1,110 / +0,183) | 1,276 (+1,232 / +0,384) | 0,852 (+0,828 / +0,048) | 1,227 (+1,213 / +0,005) | 0,899 (+0,898 / −0,020) | 1,154 (+1,115 / +0,244) |
| V2-S1 | 0,025 (+0,011 / −0,916) | 0,222 (+0,178 / −0,670) | 0,034 (+0,010 / −0,770) | 0,003 (−0,011 / −1,219) | 0,004 (+0,003 / −0,915) | 0,202 (+0,163 / −0,708) |
| V2-S2 | 0,014 (+0,000 / −0,927) | 0,044 (+0,000 / −0,848) | 0,025 (+0,001 / −0,779) | 0,015 (+0,001 / −1,207) | 0,001 (+0,000 / −0,918) | 0,044 (+0,005 / −0,866) |
| V2-S3 | 0,020 (+0,006 / −0,921) | 0,145 (+0,101 / −0,747) | 0,046 (+0,022 / −0,758) | 0,013 (−0,001 / −1,209) | 0,004 (+0,003 / −0,915) | 0,164 (+0,125 / −0,746) |
| V2-S4 | 0,020 (+0,006 / −0,921) | 0,141 (+0,097 / −0,751) | 0,046 (+0,022 / −0,758) | 0,012 (−0,002 / −1,210) | 0,004 (+0,003 / −0,915) | 0,184 (+0,145 / −0,726) |
| V2-S5 | 0,020 (+0,006 / −0,921) | 0,146 (+0,102 / −0,746) | 0,046 (+0,022 / −0,758) | 0,013 (−0,001 / −1,209) | 0,004 (+0,003 / −0,915) | 0,175 (+0,136 / −0,735) |
| V2-S6 | 0,020 (+0,006 / −0,921) | 0,145 (+0,101 / −0,747) | 0,046 (+0,022 / −0,758) | 0,012 (−0,002 / −1,210) | 0,004 (+0,003 / −0,915) | 0,175 (+0,136 / −0,735) |

**Couleurs des étoiles : erreur B−R face à Gaia (BP−RP ; plus bas = plus fidèle)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 0,218 | 0,058 | 0,021 | 0,074 | 0,148 | 0,054 |
| OLD | 0,093 (– / −0,125) | 0,046 (– / −0,012) | 0,026 (– / +0,005) | 0,053 (– / −0,021) | 0,112 (– / −0,035) | 0,060 (– / +0,006) |
| OLD-bis | 0,092 (−0,001 / −0,126) | 0,046 (+0,000 / −0,011) | 0,026 (−0,000 / +0,005) | 0,053 (+0,000 / −0,020) | 0,112 (+0,000 / −0,035) | 0,061 (+0,001 / +0,007) |
| S1 | 0,094 (+0,001 / −0,124) | 0,045 (−0,001 / −0,013) | 0,026 (−0,000 / +0,005) | 0,056 (+0,003 / −0,018) | 0,113 (+0,001 / −0,035) | 0,062 (+0,002 / +0,008) |
| S2 | 0,096 (+0,003 / −0,122) | 0,050 (+0,004 / −0,007) | 0,028 (+0,002 / +0,007) | 0,054 (+0,001 / −0,020) | 0,125 (+0,012 / −0,023) ▼ | 0,059 (−0,001 / +0,005) |
| S3 | 0,094 (+0,001 / −0,124) | 0,046 (+0,001 / −0,011) | 0,027 (+0,001 / +0,006) | 0,054 (+0,001 / −0,020) | 0,125 (+0,012 / −0,023) ▼ | 0,053 (−0,007 / −0,001) ▲ |
| S4 | 0,092 (−0,000 / −0,126) | 0,047 (+0,001 / −0,010) | 0,027 (+0,001 / +0,006) | 0,065 (+0,012 / −0,008) ▼ | 0,157 (+0,045 / +0,010) ▼ | 0,040 (−0,020 / −0,014) ▲ |
| S4b | 0,092 (−0,000 / −0,126) | 0,047 (+0,001 / −0,010) | 0,027 (+0,001 / +0,006) | 0,065 (+0,012 / −0,008) ▼ | 0,157 (+0,045 / +0,010) ▼ | 0,040 (−0,020 / −0,014) ▲ |
| S5 | 0,093 (−0,000 / −0,125) | 0,047 (+0,001 / −0,011) | 0,026 (−0,000 / +0,005) | 0,053 (−0,000 / −0,021) | 0,134 (+0,022 / −0,014) ▼ | 0,060 (−0,001 / +0,006) |
| S6 | 0,093 (+0,000 / −0,125) | 0,046 (−0,000 / −0,012) | 0,026 (+0,000 / +0,005) | 0,053 (−0,000 / −0,021) | 0,138 (+0,026 / −0,010) ▼ | 0,061 (+0,001 / +0,007) |
| S7 | 0,235 (+0,143 / +0,017) ▼ | 0,089 (+0,043 / +0,032) ▼ | 0,024 (−0,002 / +0,003) | 0,085 (+0,032 / +0,011) ▼ | 0,103 (−0,009 / −0,045) ▲ | 0,061 (+0,001 / +0,007) |
| V2 | 0,100 (+0,007 / −0,118) ▼ | 0,050 (+0,005 / −0,007) | 0,028 (+0,002 / +0,007) | 0,061 (+0,007 / −0,013) ▼ | 0,124 (+0,012 / −0,024) ▼ | 0,059 (−0,001 / +0,005) |
| V2-auto | 0,234 (+0,142 / +0,016) ▼ | 0,050 (+0,005 / −0,007) | 0,028 (+0,002 / +0,007) | 0,061 (+0,007 / −0,013) ▼ | 0,124 (+0,012 / −0,024) ▼ | 0,059 (−0,001 / +0,005) |
| V2-C | 0,234 (+0,142 / +0,016) ▼ | 0,083 (+0,038 / +0,026) ▼ | 0,025 (−0,001 / +0,004) | 0,085 (+0,032 / +0,011) ▼ | 0,148 (+0,036 / +0,000) ▼ | 0,056 (−0,004 / +0,002) |
| V2-S1 | 0,098 (+0,005 / −0,120) ▼ | 0,049 (+0,003 / −0,009) | 0,029 (+0,003 / +0,008) | 0,060 (+0,006 / −0,014) ▼ | 0,115 (+0,003 / −0,032) | 0,055 (−0,005 / +0,001) ▲ |
| V2-S2 | 0,093 (+0,000 / −0,125) | 0,049 (+0,003 / −0,009) | 0,025 (−0,001 / +0,004) | 0,057 (+0,004 / −0,017) | 0,075 (−0,037 / −0,072) ▲ | 0,059 (−0,001 / +0,005) |
| V2-S3 | 0,098 (+0,006 / −0,120) ▼ | 0,050 (+0,004 / −0,007) | 0,027 (+0,001 / +0,006) | 0,061 (+0,008 / −0,013) ▼ | 0,127 (+0,015 / −0,020) ▼ | 0,049 (−0,011 / −0,005) ▲ |
| V2-S4 | 0,100 (+0,007 / −0,118) ▼ | 0,046 (+0,000 / −0,012) | 0,028 (+0,002 / +0,007) | 0,060 (+0,007 / −0,014) ▼ | 0,126 (+0,014 / −0,022) ▼ | 0,059 (−0,001 / +0,005) |
| V2-S5 | 0,100 (+0,007 / −0,118) ▼ | 0,050 (+0,005 / −0,007) | 0,027 (+0,002 / +0,006) | 0,061 (+0,008 / −0,013) ▼ | 0,092 (−0,020 / −0,056) ▲ | 0,056 (−0,004 / +0,003) |
| V2-S6 | 0,100 (+0,008 / −0,118) ▼ | 0,050 (+0,004 / −0,008) | 0,029 (+0,003 / +0,008) | 0,062 (+0,008 / −0,012) ▼ | 0,139 (+0,027 / −0,008) ▼ | 0,046 (−0,014 / −0,008) ▲ |

**Couleurs des étoiles : erreur V−R face à Gaia (BP−RP)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 0,325 | 0,072 | 0,047 | 0,127 | 0,252 | 0,106 |
| OLD | 0,186 (– / −0,139) | 0,058 (– / −0,013) | 0,036 (– / −0,011) | 0,052 (– / −0,075) | 0,156 (– / −0,096) | 0,208 (– / +0,101) |
| OLD-bis | 0,185 (−0,001 / −0,141) | 0,058 (−0,001 / −0,014) | 0,037 (+0,000 / −0,011) | 0,052 (−0,000 / −0,075) | 0,146 (−0,009 / −0,105) | 0,206 (−0,002 / +0,100) |
| S1 | 0,184 (−0,002 / −0,141) | 0,057 (−0,002 / −0,015) | 0,035 (−0,002 / −0,012) | 0,049 (−0,003 / −0,078) | 0,181 (+0,025 / −0,071) ▼ | 0,238 (+0,030 / +0,131) ▼ |
| S2 | 0,175 (−0,011 / −0,150) ▲ | 0,055 (−0,003 / −0,016) | 0,040 (+0,003 / −0,008) | 0,057 (+0,005 / −0,071) | 0,227 (+0,071 / −0,025) ▼ | 0,198 (−0,010 / +0,092) ▲ |
| S3 | 0,194 (+0,008 / −0,132) | 0,055 (−0,004 / −0,017) | 0,037 (+0,000 / −0,011) | 0,050 (−0,002 / −0,077) | 0,237 (+0,081 / −0,015) ▼ | 0,167 (−0,040 / +0,061) ▲ |
| S4 | 0,185 (−0,001 / −0,140) | 0,058 (−0,000 / −0,013) | 0,036 (−0,001 / −0,012) | 0,046 (−0,006 / −0,081) | 0,090 (−0,066 / −0,162) ▲ | 0,074 (−0,134 / −0,032) ▲ |
| S4b | 0,185 (−0,001 / −0,140) | 0,058 (−0,000 / −0,013) | 0,036 (−0,001 / −0,012) | 0,046 (−0,006 / −0,081) | 0,090 (−0,066 / −0,162) ▲ | 0,074 (−0,134 / −0,032) ▲ |
| S5 | 0,189 (+0,003 / −0,137) | 0,059 (+0,000 / −0,013) | 0,037 (+0,000 / −0,011) | 0,052 (−0,001 / −0,076) | 0,169 (+0,013 / −0,083) ▼ | 0,203 (−0,004 / +0,097) |
| S6 | 0,186 (+0,000 / −0,139) | 0,058 (−0,001 / −0,014) | 0,037 (+0,000 / −0,010) | 0,052 (+0,000 / −0,075) | 0,157 (+0,001 / −0,095) | 0,210 (+0,003 / +0,104) |
| S7 | 0,328 (+0,142 / +0,003) ▼ | 0,071 (+0,013 / −0,000) ▼ | 0,039 (+0,002 / −0,008) | 0,279 (+0,227 / +0,152) ▼ | 0,162 (+0,006 / −0,090) | 0,174 (−0,034 / +0,068) ▲ |
| V2 | 0,184 (−0,002 / −0,142) | 0,059 (+0,000 / −0,013) | 0,033 (−0,004 / −0,014) | 0,059 (+0,007 / −0,068) | 0,115 (−0,041 / −0,137) ▲ | 0,212 (+0,005 / +0,106) |
| V2-auto | 0,323 (+0,137 / −0,003) ▼ | 0,059 (+0,000 / −0,013) | 0,033 (−0,004 / −0,014) | 0,059 (+0,007 / −0,068) | 0,115 (−0,041 / −0,137) ▲ | 0,212 (+0,005 / +0,106) |
| V2-C | 0,323 (+0,137 / −0,003) ▼ | 0,070 (+0,012 / −0,001) ▼ | 0,038 (+0,001 / −0,010) | 0,264 (+0,212 / +0,137) ▼ | 0,161 (+0,006 / −0,091) | 0,161 (−0,046 / +0,055) ▲ |
| V2-S1 | 0,179 (−0,007 / −0,146) | 0,054 (−0,005 / −0,018) | 0,045 (+0,009 / −0,002) | 0,053 (+0,001 / −0,074) | 0,147 (−0,009 / −0,105) | 0,192 (−0,016 / +0,086) ▲ |
| V2-S2 | 0,183 (−0,002 / −0,142) | 0,056 (−0,002 / −0,015) | 0,035 (−0,002 / −0,013) | 0,051 (−0,001 / −0,077) | 0,109 (−0,047 / −0,143) ▲ | 0,215 (+0,007 / +0,109) |
| V2-S3 | 0,181 (−0,005 / −0,144) | 0,058 (−0,001 / −0,014) | 0,034 (−0,003 / −0,014) | 0,060 (+0,008 / −0,067) | 0,125 (−0,030 / −0,127) ▲ | 0,142 (−0,066 / +0,035) ▲ |
| V2-S4 | 0,184 (−0,002 / −0,142) | 0,058 (−0,000 / −0,013) | 0,033 (−0,004 / −0,014) | 0,060 (+0,008 / −0,067) | 0,237 (+0,082 / −0,015) ▼ | 0,240 (+0,032 / +0,133) ▼ |
| V2-S5 | 0,181 (−0,005 / −0,144) | 0,057 (−0,002 / −0,015) | 0,033 (−0,003 / −0,014) | 0,059 (+0,007 / −0,069) | 0,128 (−0,028 / −0,124) ▲ | 0,207 (−0,001 / +0,100) |
| V2-S6 | 0,183 (−0,003 / −0,142) | 0,056 (−0,003 / −0,016) | 0,038 (+0,002 / −0,009) | 0,061 (+0,009 / −0,066) | 0,121 (−0,035 / −0,131) ▲ | 0,114 (−0,093 / +0,008) ▲ |

**Planéité du fond après dégradé (mag/arcsec² ; plus haut = plus plat)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 24,58 | 23,26 | 22,98 | 23,16 | 21,30 | 21,95 |
| OLD | 24,65 (– / +0,07) | 23,39 (– / +0,12) | 23,20 (– / +0,22) | 23,20 (– / +0,04) | 20,97 (– / −0,33) | 21,22 (– / −0,73) |
| OLD-bis | 24,65 (−0,00 / +0,07) | 23,38 (−0,01 / +0,11) | 23,20 (+0,00 / +0,22) | 23,20 (−0,00 / +0,04) | 20,97 (+0,00 / −0,33) | 21,22 (−0,00 / −0,73) |
| S1 | 24,71 (+0,06 / +0,13) | 23,45 (+0,07 / +0,19) | 23,15 (−0,05 / +0,17) | 23,26 (+0,06 / +0,10) | 21,04 (+0,08 / −0,26) | 21,23 (+0,01 / −0,71) |
| S2 | 24,69 (+0,04 / +0,11) | 23,31 (−0,08 / +0,05) | 23,13 (−0,07 / +0,15) | 23,20 (−0,01 / +0,03) | 20,95 (−0,02 / −0,35) | 21,30 (+0,08 / −0,64) |
| S3 | 24,63 (−0,02 / +0,05) | 23,37 (−0,02 / +0,10) | 23,20 (+0,00 / +0,22) | 23,19 (−0,01 / +0,03) | 20,99 (+0,02 / −0,31) | 21,34 (+0,12 / −0,61) ▲ |
| S4 | 24,65 (+0,00 / +0,07) | 23,36 (−0,03 / +0,09) | 23,25 (+0,06 / +0,27) | 23,21 (+0,01 / +0,05) | 20,97 (+0,00 / −0,33) | 21,86 (+0,64 / −0,09) ▲ |
| S4b | 24,65 (+0,00 / +0,07) | 23,36 (−0,03 / +0,09) | 23,25 (+0,06 / +0,27) | 23,21 (+0,01 / +0,05) | 20,97 (+0,00 / −0,33) | 21,86 (+0,64 / −0,09) ▲ |
| S5 | 24,65 (−0,00 / +0,06) | 23,38 (−0,00 / +0,12) | 23,16 (−0,03 / +0,18) | 23,20 (−0,00 / +0,04) | 20,97 (+0,00 / −0,33) | 21,24 (+0,02 / −0,71) |
| S6 | 24,64 (−0,01 / +0,05) | 23,39 (+0,01 / +0,13) | 23,20 (+0,00 / +0,22) | 23,22 (+0,02 / +0,06) | 20,96 (−0,00 / −0,34) | 21,20 (−0,02 / −0,75) |
| S7 | 23,68 (−0,97 / −0,90) ▼ | 23,56 (+0,17 / +0,29) ▲ | 23,22 (+0,03 / +0,24) | 22,90 (−0,30 / −0,26) ▼ | 21,36 (+0,39 / +0,06) ▲ | 22,12 (+0,90 / +0,17) ▲ |
| V2 | 24,70 (+0,05 / +0,11) | 23,41 (+0,03 / +0,15) | 23,12 (−0,08 / +0,14) | 23,28 (+0,08 / +0,12) | 21,03 (+0,07 / −0,26) | 21,23 (+0,01 / −0,72) |
| V2-auto | 23,69 (−0,96 / −0,89) ▼ | 23,41 (+0,03 / +0,15) | 23,12 (−0,08 / +0,14) | 23,28 (+0,08 / +0,12) | 21,03 (+0,07 / −0,26) | 21,23 (+0,01 / −0,72) |
| V2-C | 23,69 (−0,96 / −0,89) ▼ | 23,54 (+0,15 / +0,27) ▲ | 23,21 (+0,01 / +0,23) | 22,91 (−0,29 / −0,25) ▼ | 21,28 (+0,32 / −0,01) ▲ | 22,11 (+0,89 / +0,16) ▲ |
| V2-S1 | 24,70 (+0,05 / +0,12) | 23,30 (−0,09 / +0,03) | 23,07 (−0,12 / +0,09) ▼ | 23,18 (−0,02 / +0,02) | 21,16 (+0,20 / −0,14) ▲ | 21,15 (−0,07 / −0,80) |
| V2-S2 | 24,71 (+0,06 / +0,13) | 23,42 (+0,04 / +0,16) | 23,13 (−0,06 / +0,16) | 23,28 (+0,08 / +0,12) | 21,01 (+0,04 / −0,29) | 21,23 (+0,01 / −0,72) |
| V2-S3 | 24,70 (+0,05 / +0,12) | 23,41 (+0,03 / +0,15) | 23,14 (−0,06 / +0,16) | 23,29 (+0,09 / +0,13) | 21,03 (+0,06 / −0,27) | 21,66 (+0,44 / −0,28) ▲ |
| V2-S4 | 24,70 (+0,05 / +0,11) | 23,47 (+0,08 / +0,20) | 23,12 (−0,08 / +0,14) | 23,26 (+0,06 / +0,10) | 21,05 (+0,08 / −0,25) | 21,27 (+0,05 / −0,68) |
| V2-S5 | 24,69 (+0,04 / +0,11) | 23,42 (+0,03 / +0,15) | 23,17 (−0,03 / +0,19) | 23,29 (+0,09 / +0,13) | 21,03 (+0,06 / −0,27) | 21,24 (+0,02 / −0,71) |
| V2-S6 | 24,69 (+0,04 / +0,11) | 23,40 (+0,02 / +0,14) | 23,12 (−0,08 / +0,14) | 23,30 (+0,10 / +0,14) | 21,04 (+0,07 / −0,26) | 21,71 (+0,49 / −0,24) ▲ |

**Traînées le long de la dérive (« walking noise » ; 1 = aucune)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 1,00 | 1,00 | 1,01 | 1,00 | 0,96 | 1,00 |
| OLD | 1,00 (– / +0,00) | 0,98 (– / −0,02) | 1,04 (– / +0,03) | 0,99 (– / −0,01) | 0,96 (– / −0,00) | 1,00 (– / −0,00) |
| OLD-bis | 1,00 (−0,00 / +0,00) | 0,98 (−0,00 / −0,02) | 1,04 (−0,00 / +0,03) | 0,99 (−0,00 / −0,01) | 0,96 (−0,00 / −0,00) | 1,00 (+0,00 / −0,00) |
| S1 | 1,00 (+0,00 / +0,00) | 0,99 (+0,01 / −0,01) | 1,04 (+0,00 / +0,03) | 0,99 (+0,00 / −0,01) | 0,95 (−0,01 / −0,01) | 1,00 (−0,00 / −0,00) |
| S2 | 1,00 (−0,00 / +0,00) | 0,98 (+0,00 / −0,02) | 1,04 (−0,00 / +0,03) | 0,98 (−0,00 / −0,02) | 0,96 (+0,00 / −0,00) | 1,00 (−0,00 / −0,00) |
| S3 | 1,00 (+0,00 / +0,00) | 0,99 (+0,01 / −0,01) | 1,04 (−0,00 / +0,02) | 0,99 (+0,00 / −0,01) | 0,97 (+0,01 / +0,01) | 1,00 (+0,00 / −0,00) |
| S4 | 1,00 (−0,00 / +0,00) | 0,99 (+0,01 / −0,01) | 0,99 (−0,06 / −0,03) ▲ | 0,99 (+0,00 / −0,01) | 0,99 (+0,04 / +0,03) ▼ | 1,01 (+0,01 / +0,01) |
| S4b | 1,00 (−0,00 / +0,00) | 0,99 (+0,01 / −0,01) | 0,99 (−0,06 / −0,03) ▲ | 0,99 (+0,00 / −0,01) | 0,99 (+0,04 / +0,03) ▼ | 1,01 (+0,01 / +0,01) |
| S5 | 1,00 (−0,00 / −0,00) | 0,98 (+0,00 / −0,02) | 1,04 (−0,00 / +0,03) | 0,99 (−0,00 / −0,02) | 0,97 (+0,01 / +0,01) | 1,00 (−0,00 / −0,00) |
| S6 | 1,00 (+0,00 / +0,00) | 0,98 (−0,00 / −0,02) | 1,04 (−0,00 / +0,03) | 0,99 (−0,00 / −0,02) | 0,95 (−0,00 / −0,01) | 1,00 (+0,01 / +0,00) |
| S7 | 1,00 (−0,00 / +0,00) | 0,99 (+0,01 / −0,01) | 1,00 (−0,04 / −0,01) ▲ | 0,99 (+0,01 / −0,01) | 0,98 (+0,02 / +0,02) | 0,99 (−0,01 / −0,01) |
| V2 | 1,00 (−0,00 / −0,00) | 0,99 (+0,01 / −0,01) | 1,04 (−0,00 / +0,03) | 0,99 (+0,00 / −0,01) | 0,95 (−0,01 / −0,01) | 1,01 (+0,01 / +0,01) |
| V2-auto | 1,00 (−0,00 / −0,00) | 0,99 (+0,01 / −0,01) | 1,04 (−0,00 / +0,03) | 0,99 (+0,00 / −0,01) | 0,95 (−0,01 / −0,01) | 1,01 (+0,01 / +0,01) |
| V2-C | 1,00 (−0,00 / −0,00) | 0,98 (+0,00 / −0,02) | 1,04 (−0,00 / +0,03) | 0,99 (+0,01 / −0,01) | 1,00 (+0,04 / +0,04) ▼ | 0,99 (−0,01 / −0,01) |
| V2-S1 | 1,00 (−0,00 / −0,00) | 0,99 (+0,01 / −0,01) | 0,99 (−0,06 / −0,03) ▲ | 0,99 (+0,01 / −0,01) | 1,00 (+0,04 / +0,03) ▼ | 1,00 (+0,00 / −0,00) |
| V2-S2 | 1,00 (−0,00 / −0,00) | 0,99 (+0,02 / −0,01) | 1,04 (−0,00 / +0,03) | 0,99 (+0,00 / −0,01) | 0,95 (−0,01 / −0,02) | 1,00 (+0,01 / +0,00) |
| V2-S3 | 1,00 (−0,00 / −0,00) | 0,99 (+0,01 / −0,01) | 1,05 (+0,00 / +0,03) | 0,99 (+0,00 / −0,01) | 0,95 (−0,01 / −0,01) | 1,01 (+0,01 / +0,01) |
| V2-S4 | 1,00 (−0,00 / −0,00) | 0,99 (+0,02 / −0,01) | 1,04 (−0,00 / +0,03) | 0,99 (+0,00 / −0,01) | 0,96 (−0,00 / −0,00) | 1,00 (−0,00 / −0,00) |
| V2-S5 | 1,00 (+0,00 / +0,00) | 0,99 (+0,01 / −0,01) | 1,04 (−0,00 / +0,03) | 0,99 (+0,00 / −0,01) | 0,96 (+0,00 / −0,00) | 1,01 (+0,01 / +0,01) |
| V2-S6 | 1,00 (−0,00 / −0,00) | 1,00 (+0,02 / −0,00) | 1,04 (+0,00 / +0,03) | 0,99 (+0,00 / −0,01) | 0,96 (+0,00 / −0,00) | 1,00 (+0,00 / −0,00) |

**Poses gardées**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 30 | 30 | 30 | 15 | 29 | 30 |
| OLD | 30 (– / +0) | 30 (– / +0) | 30 (– / +0) | 15 (– / +0) | 30 (– / +1) | 16 (– / −14) |
| OLD-bis | 30 (+0 / +0) | 30 (+0 / +0) | 30 (+0 / +0) | 15 (+0 / +0) | 30 (+0 / +1) | 16 (+0 / −14) |
| S1 | 30 (+0 / +0) | 30 (+0 / +0) | 30 (+0 / +0) | 15 (+0 / +0) | 30 (+0 / +1) | 16 (+0 / −14) |
| S2 | 30 (+0 / +0) | 30 (+0 / +0) | 30 (+0 / +0) | 15 (+0 / +0) | 30 (+0 / +1) | 16 (+0 / −14) |
| S3 | 30 (+0 / +0) | 30 (+0 / +0) | 30 (+0 / +0) | 15 (+0 / +0) | 30 (+0 / +1) | 16 (+0 / −14) |
| S4 | 30 (+0 / +0) | 28 (−2 / −2) | 23 (−7 / −7) | 10 (−5 / −5) | 24 (−6 / −5) | 23 (+7 / −7) |
| S4b | 30 (+0 / +0) | 28 (−2 / −2) | 23 (−7 / −7) | 10 (−5 / −5) | 24 (−6 / −5) | 23 (+7 / −7) |
| S5 | 30 (+0 / +0) | 30 (+0 / +0) | 30 (+0 / +0) | 15 (+0 / +0) | 30 (+0 / +1) | 16 (+0 / −14) |
| S6 | 30 (+0 / +0) | 30 (+0 / +0) | 30 (+0 / +0) | 15 (+0 / +0) | 30 (+0 / +1) | 16 (+0 / −14) |
| S7 | 30 (+0 / +0) | 30 (+0 / +0) | 30 (+0 / +0) | 15 (+0 / +0) | 30 (+0 / +1) | 16 (+0 / −14) |
| V2 | 30 (+0 / +0) | 28 (−2 / −2) | 30 (+0 / +0) | 14 (−1 / −1) | 28 (−2 / −1) | 23 (+7 / −7) |
| V2-auto | 30 (+0 / +0) | 28 (−2 / −2) | 30 (+0 / +0) | 14 (−1 / −1) | 28 (−2 / −1) | 23 (+7 / −7) |
| V2-C | 30 (+0 / +0) | 28 (−2 / −2) | 30 (+0 / +0) | 14 (−1 / −1) | 28 (−2 / −1) | 23 (+7 / −7) |
| V2-S1 | 30 (+0 / +0) | 28 (−2 / −2) | 23 (−7 / −7) | 10 (−5 / −5) | 24 (−6 / −5) | 23 (+7 / −7) |
| V2-S2 | 30 (+0 / +0) | 28 (−2 / −2) | 30 (+0 / +0) | 14 (−1 / −1) | 28 (−2 / −1) | 23 (+7 / −7) |
| V2-S3 | 30 (+0 / +0) | 28 (−2 / −2) | 30 (+0 / +0) | 14 (−1 / −1) | 28 (−2 / −1) | 24 (+8 / −6) |
| V2-S4 | 30 (+0 / +0) | 30 (+0 / +0) | 30 (+0 / +0) | 15 (+0 / +0) | 30 (+0 / +1) | 16 (+0 / −14) |
| V2-S5 | 30 (+0 / +0) | 28 (−2 / −2) | 30 (+0 / +0) | 14 (−1 / −1) | 28 (−2 / −1) | 23 (+7 / −7) |
| V2-S6 | 30 (+0 / +0) | 28 (−2 / −2) | 30 (+0 / +0) | 14 (−1 / −1) | 28 (−2 / −1) | 23 (+7 / −7) |

**Durée de l'empilement seul (min ; machine partagée)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | – | – | – | – | – | – |
| OLD | 0,8 (– / –) | 0,7 (– / –) | 1,3 (– / –) | 3,7 (– / –) | 0,1 (– / –) | 0,6 (– / –) |
| OLD-bis | 1,5 (+0,8 / –) | 0,7 (+0,0 / –) | 1,3 (+0,0 / –) | 3,5 (−0,3 / –) | 0,3 (+0,2 / –) | 1,0 (+0,4 / –) |
| S1 | 1,9 (+1,2 / –) | 0,9 (+0,2 / –) | 1,5 (+0,2 / –) | 5,0 (+1,3 / –) | 0,2 (+0,0 / –) | 0,9 (+0,3 / –) |
| S2 | 0,8 (+0,1 / –) | 0,8 (+0,0 / –) | 1,9 (+0,7 / –) | 9,4 (+5,6 / –) | 0,2 (+0,0 / –) | 0,9 (+0,4 / –) |
| S3 | 0,8 (+0,1 / –) | 0,8 (+0,1 / –) | 1,4 (+0,1 / –) | 2,5 (−1,3 / –) | 0,2 (+0,0 / –) | 0,9 (+0,4 / –) |
| S4 | 0,8 (+0,1 / –) | 0,8 (+0,0 / –) | 1,1 (−0,2 / –) | 2,4 (−1,3 / –) | 0,2 (+0,1 / –) | 1,4 (+0,8 / –) |
| S4b | 1,3 (+0,6 / –) | 0,7 (−0,0 / –) | 1,1 (−0,2 / –) | 4,4 (+0,6 / –) | 0,1 (+0,0 / –) | 1,0 (+0,5 / –) |
| S5 | 2,2 (+1,4 / –) | 1,0 (+0,3 / –) | 1,8 (+0,6 / –) | 4,3 (+0,5 / –) | 0,2 (+0,1 / –) | 1,4 (+0,9 / –) |
| S6 | 1,5 (+0,7 / –) | 0,9 (+0,2 / –) | 2,2 (+0,9 / –) | 2,5 (−1,3 / –) | 0,2 (+0,0 / –) | 0,6 (−0,0 / –) |
| S7 | 2,1 (+1,3 / –) | 2,0 (+1,2 / –) | 3,9 (+2,7 / –) | 4,8 (+1,1 / –) | 0,3 (+0,2 / –) | 1,5 (+0,9 / –) |
| V2 | 1,3 (+0,5 / –) | 0,8 (+0,1 / –) | 2,9 (+1,6 / –) | 4,2 (+0,5 / –) | 0,2 (+0,1 / –) | 1,2 (+0,6 / –) |
| V2-auto | 4,4 (+3,6 / –) | 0,8 (+0,1 / –) | 2,9 (+1,6 / –) | 4,2 (+0,5 / –) | 0,2 (+0,1 / –) | 1,2 (+0,6 / –) |
| V2-C | 4,4 (+3,6 / –) | 4,3 (+3,6 / –) | 3,6 (+2,3 / –) | 4,8 (+1,1 / –) | 4,6 (+4,5 / –) | 1,9 (+1,3 / –) |
| V2-S1 | 1,4 (+0,7 / –) | 0,8 (+0,0 / –) | 2,4 (+1,2 / –) | 2,2 (−1,5 / –) | 0,4 (+0,3 / –) | 1,2 (+0,7 / –) |
| V2-S2 | 2,0 (+1,2 / –) | 1,1 (+0,4 / –) | 2,0 (+0,8 / –) | 5,7 (+2,0 / –) | 0,4 (+0,2 / –) | 2,1 (+1,5 / –) |
| V2-S3 | 1,7 (+0,9 / –) | 0,8 (+0,1 / –) | 1,6 (+0,4 / –) | 3,3 (−0,4 / –) | 0,3 (+0,1 / –) | 1,1 (+0,6 / –) |
| V2-S4 | 1,5 (+0,7 / –) | 1,0 (+0,2 / –) | 1,6 (+0,3 / –) | 3,7 (−0,0 / –) | 0,2 (+0,1 / –) | 0,7 (+0,1 / –) |
| V2-S5 | 1,1 (+0,3 / –) | 0,6 (−0,1 / –) | 1,2 (−0,0 / –) | 2,4 (−1,3 / –) | 0,1 (−0,0 / –) | 1,0 (+0,5 / –) |
| V2-S6 | 1,7 (+1,0 / –) | 1,0 (+0,3 / –) | 2,0 (+0,7 / –) | 4,9 (+1,1 / –) | 0,2 (+0,1 / –) | 1,1 (+0,6 / –) |

**Durée calibration + alignement + empilement (min ; Siril : conversion comprise, 8 cœurs)**

| | heart | iris | m51 | m8 | seestar | rosette |
|---|---|---|---|---|---|---|
| Siril | 2,0 | 2,8 | 2,4 | 3,5 | 0,2 | 1,3 |
| OLD | 2,3 (– / +0,3) | 2,3 (– / −0,5) | 3,6 (– / +1,2) | 6,8 (– / +3,3) | 0,8 (– / +0,6) | 2,5 (– / +1,2) |
| OLD-bis | 3,1 (+0,8 / +1,0) | 2,3 (−0,0 / −0,5) | 3,7 (+0,1 / +1,3) | 6,5 (−0,3 / +3,0) | 1,0 (+0,2 / +0,8) | 2,8 (+0,3 / +1,5) |
| S1 | 3,5 (+1,2 / +1,5) | 2,5 (+0,2 / −0,3) | 4,0 (+0,3 / +1,6) | 8,1 (+1,3 / +4,6) | 0,9 (+0,0 / +0,6) | 2,8 (+0,3 / +1,4) |
| S2 | 2,3 (+0,1 / +0,3) | 2,3 (+0,0 / −0,5) | 4,3 (+0,7 / +1,9) | 12,4 (+5,6 / +8,9) | 0,9 (+0,0 / +0,6) | 2,9 (+0,4 / +1,5) |
| S3 | 2,4 (+0,1 / +0,3) | 2,4 (+0,1 / −0,4) | 3,7 (+0,1 / +1,4) | 5,5 (−1,3 / +2,0) | 0,9 (+0,0 / +0,6) | 2,9 (+0,4 / +1,6) |
| S4 | 2,4 (+0,1 / +0,3) | 2,3 (+0,0 / −0,5) | 3,5 (−0,2 / +1,1) | 5,5 (−1,3 / +2,0) | 0,9 (+0,1 / +0,7) | 3,3 (+0,8 / +2,0) |
| S4b | 2,9 (+0,6 / +0,8) | 2,3 (−0,0 / −0,5) | 3,4 (−0,2 / +1,1) | 7,4 (+0,6 / +3,9) | 0,8 (+0,0 / +0,6) | 3,0 (+0,5 / +1,7) |
| S5 | 3,7 (+1,4 / +1,7) | 2,6 (+0,3 / −0,2) | 4,2 (+0,6 / +1,8) | 7,3 (+0,5 / +3,8) | 1,0 (+0,1 / +0,7) | 3,4 (+0,9 / +2,0) |
| S6 | 3,1 (+0,8 / +1,0) | 2,7 (+0,4 / −0,1) | 4,5 (+0,9 / +2,2) | 5,6 (−1,2 / +2,0) | 0,9 (+0,1 / +0,7) | 2,4 (−0,1 / +1,1) |
| S7 | 3,6 (+1,3 / +1,6) | 3,6 (+1,2 / +0,7) | 6,3 (+2,7 / +3,9) | 7,9 (+1,1 / +4,4) | 1,0 (+0,2 / +0,8) | 3,5 (+0,9 / +2,1) |
| V2 | 2,9 (+0,6 / +0,9) | 2,4 (+0,1 / −0,4) | 5,4 (+1,7 / +3,0) | 7,5 (+0,7 / +4,0) | 0,9 (+0,0 / +0,6) | 3,1 (+0,6 / +1,7) |
| V2-auto | 6,0 (+3,7 / +4,0) | 2,4 (+0,1 / −0,4) | 5,4 (+1,7 / +3,0) | 7,5 (+0,7 / +4,0) | 0,9 (+0,0 / +0,6) | 3,1 (+0,6 / +1,7) |
| V2-C | 6,0 (+3,7 / +4,0) | 5,9 (+3,6 / +3,1) | 6,0 (+2,4 / +3,6) | 8,1 (+1,3 / +4,6) | 5,3 (+4,4 / +5,0) | 3,8 (+1,2 / +2,4) |
| V2-S1 | 3,0 (+0,7 / +1,0) | 2,6 (+0,3 / −0,3) | 4,8 (+1,2 / +2,4) | 5,3 (−1,5 / +1,8) | 1,2 (+0,3 / +0,9) | 3,1 (+0,6 / +1,8) |
| V2-S2 | 3,6 (+1,3 / +1,5) | 2,7 (+0,4 / −0,2) | 4,5 (+0,9 / +2,1) | 9,0 (+2,2 / +5,5) | 1,0 (+0,2 / +0,8) | 4,0 (+1,5 / +2,7) |
| V2-S3 | 3,3 (+1,0 / +1,3) | 2,4 (+0,1 / −0,4) | 4,1 (+0,4 / +1,7) | 6,6 (−0,2 / +3,1) | 0,9 (+0,1 / +0,7) | 3,0 (+0,5 / +1,7) |
| V2-S4 | 3,1 (+0,8 / +1,1) | 2,5 (+0,2 / −0,3) | 4,1 (+0,4 / +1,7) | 7,0 (+0,2 / +3,5) | 0,9 (+0,0 / +0,6) | 2,6 (+0,0 / +1,2) |
| V2-S5 | 2,7 (+0,4 / +0,7) | 2,2 (−0,1 / −0,7) | 3,7 (+0,0 / +1,3) | 5,7 (−1,1 / +2,1) | 0,8 (−0,0 / +0,5) | 2,9 (+0,4 / +1,6) |
| V2-S6 | 3,3 (+1,0 / +1,3) | 2,6 (+0,3 / −0,2) | 4,4 (+0,8 / +2,0) | 7,9 (+1,2 / +4,4) | 0,9 (+0,1 / +0,7) | 3,0 (+0,5 / +1,7) |
<!-- /crossed -->

</details>

### Les tableaux complets (Base)

<details><summary>Ouvrir les tableaux (Base : OLD, S1, V2)</summary>

<!-- crossed:base -->
**Profondeur : étoile détectable à 5σ (magnitude G ; plus haut = plus profond)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 16,61 | 18,09 | 18,78 | 16,56 | 17,72 |
| OLD | 16,66 (– / +0,04) | 17,94 (– / −0,15) | 18,72 (– / −0,07) | 16,70 (– / +0,14) | 17,68 (– / −0,04) |
| S1 | 16,69 (+0,03 / +0,08) | 17,96 (+0,02 / −0,13) | 18,75 (+0,03 / −0,04) | 16,70 (+0,00 / +0,14) | 17,60 (−0,08 / −0,12) ▼ |
| V2 | 16,65 (−0,01 / +0,04) | 17,96 (+0,03 / −0,12) | 18,73 (+0,01 / −0,06) | 16,66 (−0,04 / +0,10) | 17,66 (−0,02 / −0,05) |

**Bruit du fond, petite échelle (mag/arcsec² ; plus haut = moins de bruit)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 24,82 | 24,54 | 25,30 | 23,20 | 24,24 |
| OLD | 24,73 (– / −0,09) | 24,38 (– / −0,17) | 25,19 (– / −0,11) | 23,15 (– / −0,05) | 24,09 (– / −0,15) |
| S1 | 24,73 (+0,00 / −0,09) | 24,39 (+0,02 / −0,15) | 25,22 (+0,03 / −0,08) | 23,15 (+0,00 / −0,05) | 24,11 (+0,02 / −0,13) |
| V2 | 24,72 (−0,00 / −0,09) | 24,38 (+0,01 / −0,16) | 25,18 (−0,02 / −0,12) | 23,09 (−0,06 / −0,11) ▼ | 24,19 (+0,09 / −0,05) ▲ |

**Bruit du fond, échelle 4 fois plus grande (mag/arcsec²)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 25,28 | 25,84 | 26,48 | 24,25 | 25,27 |
| OLD | 25,35 (– / +0,06) | 25,65 (– / −0,20) | 26,45 (– / −0,04) | 24,34 (– / +0,09) | 25,12 (– / −0,15) |
| S1 | 25,35 (+0,00 / +0,06) | 25,71 (+0,06 / −0,14) ▲ | 26,46 (+0,01 / −0,03) | 24,34 (+0,00 / +0,09) | 25,19 (+0,07 / −0,08) ▲ |
| V2 | 25,33 (−0,01 / +0,05) | 25,71 (+0,06 / −0,14) ▲ | 26,42 (−0,03 / −0,07) | 24,35 (+0,01 / +0,09) | 25,25 (+0,13 / −0,02) ▲ |

**Signal/bruit sur la cible**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 6,4 | 4,2 | 3,4 | 36,7 | 8,3 |
| OLD | 6,3 (– / −2 %) | 3,5 (– / −15 %) | 3,0 (– / −12 %) | 36,6 (– / −0 %) | 7,3 (– / −12 %) |
| S1 | 6,3 (+0 % / −2 %) | 3,7 (+4 % / −12 %) ▲ | 3,1 (+3 % / −10 %) | 36,6 (+0 % / −0 %) | 7,5 (+3 % / −9 %) |
| V2 | 6,1 (−3 % / −5 %) ▼ | 3,8 (+8 % / −8 %) ▲ | 2,7 (−10 % / −21 %) ▼ | 31,8 (−13 % / −13 %) ▼ | 8,2 (+12 % / −1 %) ▲ |

**Étoiles faibles trouvées à 5σ (les mêmes étoiles Gaia pour toutes)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 24397 | 6583 | 1626 | 1029 | 6799 |
| OLD | 24705 (– / +1 %) | 6428 (– / −2 %) | 1553 (– / −4 %) | 1065 (– / +3 %) | 6759 (– / −1 %) |
| S1 | 24718 (+0 % / +1 %) | 6416 (−0 % / −3 %) | 1567 (+1 % / −4 %) | 1065 (+0 % / +3 %) | 6695 (−1 % / −2 %) |
| V2 | 24613 (−0 % / +1 %) | 6418 (−0 % / −3 %) | 1559 (+0 % / −4 %) | 1048 (−2 % / +2 %) | 6809 (+1 % / +0 %) |

**Finesse des étoiles : FWHM au centre (″ ; plus bas = plus fines)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 13,28 | 6,05 | 7,03 | 5,72 | 5,60 |
| OLD | 11,65 (– / −12 %) | 5,96 (– / −1 %) | 7,06 (– / +0 %) | 5,70 (– / −0 %) | 5,70 (– / +2 %) |
| S1 | 11,65 (+0 % / −12 %) | 6,16 (+3 % / +2 %) ▼ | 7,09 (+1 % / +1 %) | 5,70 (+0 % / −0 %) | 5,91 (+4 % / +6 %) ▼ |
| V2 | 11,75 (+1 % / −12 %) | 6,21 (+4 % / +3 %) ▼ | 7,06 (+0 % / +1 %) | 5,61 (−1 % / −2 %) | 5,87 (+3 % / +5 %) ▼ |

**Rondeur des étoiles au centre (1 = rondes)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 0,92 | 0,90 | 0,81 | 0,91 | 0,78 |
| OLD | 0,88 (– / −0,04) | 0,89 (– / −0,01) | 0,83 (– / +0,01) | 0,87 (– / −0,05) | 0,77 (– / −0,01) |
| S1 | 0,88 (−0,00 / −0,04) | 0,89 (+0,01 / −0,00) | 0,83 (+0,00 / +0,01) | 0,87 (+0,00 / −0,05) | 0,79 (+0,02 / +0,01) |
| V2 | 0,88 (−0,00 / −0,04) | 0,89 (+0,01 / −0,01) | 0,83 (+0,01 / +0,02) | 0,89 (+0,02 / −0,03) | 0,78 (+0,02 / +0,01) |

**Points d'une seule couleur (pixels chauds restants, par degré²)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 1 | 3 | 5 | 0 | 3 |
| OLD | 126 (– / +125) | 19 (– / +16) | 27 (– / +22) | 9 (– / +9) | 17 (– / +14) |
| S1 | 125 (−1 / +125) | 15 (−4 / +11) ▲ | 26 (−1 / +21) | 9 (+0 / +9) | 10 (−8 / +7) ▲ |
| V2 | 144 (+18 / +143) | 20 (+1 / +16) | 23 (−4 / +19) | 13 (+4 / +13) ▼ | 18 (+1 / +15) |

**Taux de rejet (% des échantillons ; repère de Siril : 0,1 à 0,5 %)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 0,760 | 0,722 | 0,629 | 0,718 | 0,738 |
| OLD | 0,021 (– / −0,739) | 0,050 (– / −0,672) | 0,030 (– / −0,599) | 0,001 (– / −0,717) | 0,042 (– / −0,696) |
| S1 | 0,021 (+0,000 / −0,739) | 0,053 (+0,003 / −0,669) | 0,030 (+0,000 / −0,599) | 0,001 (+0,000 / −0,717) | 0,047 (+0,005 / −0,691) |
| V2 | 0,014 (−0,007 / −0,746) | 0,154 (+0,104 / −0,568) | 0,050 (+0,020 / −0,579) | 0,004 (+0,003 / −0,714) | 0,134 (+0,092 / −0,604) |

**Couleurs des étoiles : erreur B−R face à Gaia (BP−RP ; plus bas = plus fidèle)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 0,173 | 0,050 | 0,018 | 0,075 | 0,041 |
| OLD | 0,062 (– / −0,111) | 0,038 (– / −0,012) | 0,026 (– / +0,008) | 0,048 (– / −0,027) | 0,034 (– / −0,007) |
| S1 | 0,063 (+0,000 / −0,110) | 0,038 (+0,000 / −0,011) | 0,027 (+0,001 / +0,009) | 0,048 (+0,000 / −0,027) | 0,046 (+0,012 / +0,005) ▼ |
| V2 | 0,064 (+0,002 / −0,109) | 0,038 (+0,001 / −0,011) | 0,026 (+0,000 / +0,009) | 0,054 (+0,006 / −0,021) ▼ | 0,045 (+0,012 / +0,004) ▼ |

**Couleurs des étoiles : erreur V−R face à Gaia (BP−RP)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 0,254 | 0,050 | 0,040 | 0,226 | 0,089 |
| OLD | 0,099 (– / −0,155) | 0,039 (– / −0,011) | 0,029 (– / −0,012) | 0,056 (– / −0,170) | 0,050 (– / −0,039) |
| S1 | 0,100 (+0,000 / −0,155) | 0,039 (+0,000 / −0,011) | 0,028 (−0,000 / −0,012) | 0,056 (+0,000 / −0,170) | 0,135 (+0,085 / +0,046) ▼ |
| V2 | 0,098 (−0,002 / −0,156) | 0,040 (+0,001 / −0,010) | 0,027 (−0,001 / −0,013) | 0,061 (+0,005 / −0,165) | 0,132 (+0,082 / +0,043) ▼ |

**Planéité du fond après dégradé (mag/arcsec² ; plus haut = plus plat)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 24,73 | 23,65 | 23,00 | 21,51 | 22,00 |
| OLD | 24,80 (– / +0,07) | 23,42 (– / −0,23) | 23,17 (– / +0,17) | 21,25 (– / −0,26) | 21,86 (– / −0,13) |
| S1 | 24,80 (−0,00 / +0,07) | 23,62 (+0,20 / −0,03) ▲ | 23,18 (+0,01 / +0,18) | 21,25 (+0,00 / −0,26) | 21,52 (−0,34 / −0,48) ▼ |
| V2 | 24,81 (+0,01 / +0,08) | 23,62 (+0,20 / −0,03) ▲ | 23,20 (+0,03 / +0,20) | 21,24 (−0,01 / −0,27) | 21,59 (−0,28 / −0,41) ▼ |

**Traînées le long de la dérive (« walking noise » ; 1 = aucune)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 1,00 | 1,01 | 1,06 | 1,05 | 1,04 |
| OLD | 1,00 (– / +0,00) | 0,99 (– / −0,02) | 1,04 (– / −0,01) | 1,05 (– / +0,01) | 1,03 (– / −0,00) |
| S1 | 1,00 (−0,00 / +0,00) | 0,98 (−0,01 / −0,03) | 1,04 (+0,00 / −0,01) | 1,05 (+0,00 / +0,01) | 1,05 (+0,01 / +0,01) |
| V2 | 1,00 (−0,00 / −0,00) | 0,99 (−0,00 / −0,02) | 1,03 (−0,01 / −0,03) | 1,04 (−0,01 / −0,00) | 1,06 (+0,03 / +0,02) |

**Poses gardées**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 100 | 100 | 100 | 93 | 100 |
| OLD | 99 (– / −1) | 99 (– / −1) | 100 (– / +0) | 100 (– / +7) | 56 (– / −44) |
| S1 | 99 (+0 / −1) | 99 (+0 / −1) | 100 (+0 / +0) | 100 (+0 / +7) | 56 (+0 / −44) |
| V2 | 94 (−5 / −6) | 97 (−2 / −3) | 91 (−9 / −9) | 88 (−12 / −5) | 90 (+34 / −10) |

**Durée de l'empilement seul (min ; machine partagée)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | – | – | – | – | – |
| OLD | 2,9 (– / –) | 2,9 (– / –) | 3,8 (– / –) | 4,1 (– / –) | 1,8 (– / –) |
| S1 | 2,5 (−0,4 / –) | 2,1 (−0,7 / –) | 4,5 (+0,6 / –) | 0,6 (−3,5 / –) | 1,2 (−0,6 / –) |
| V2 | 2,0 (−0,8 / –) | 2,0 (−0,8 / –) | 2,9 (−1,0 / –) | 0,6 (−3,5 / –) | 1,9 (+0,1 / –) |

**Durée calibration + alignement + empilement (min ; Siril : conversion comprise, 8 cœurs)**

| | heart | iris | m51 | seestar | rosette |
|---|---|---|---|---|---|
| Siril | 5,3 | 6,2 | 9,1 | 0,8 | 4,2 |
| OLD | 4,9 (– / −0,4) | 4,8 (– / −1,4) | 6,4 (– / −2,8) | 4,8 (– / +4,0) | 3,7 (– / −0,5) |
| S1 | 4,6 (−0,3 / −0,8) | 4,1 (−0,6 / −2,0) | 7,3 (+0,9 / −1,8) | 1,2 (−3,6 / +0,4) | 3,2 (−0,5 / −1,0) |
| V2 | 4,2 (−0,7 / −1,2) | 4,1 (−0,7 / −2,1) | 5,7 (−0,7 / −3,4) | 1,3 (−3,5 / +0,5) | 3,9 (+0,2 / −0,3) |
<!-- /crossed -->

</details>

## Le drizzle de Siril (2 novembre)

**La question.** Notre drizzle (Bayer) laisse plus de bruit de fond que l'empilement classique de Siril. Est-ce un
défaut de notre drizzle, ou du drizzle en général ? On a fait tourner le drizzle de Siril lui-même sur les mêmes poses.

**Réponse.**

- **Le même bruit en plus apparaît chez Siril.** Entre l'empilement classique et le drizzle, le bruit de fond fin perd
  en moyenne 0,17 mag/arcsec² chez Siril (de 0,06 à 0,36 selon le cas) et 0,12 chez nous (de 0,04 à 0,17) ; à grande
  échelle, 0,16 chez Siril et 0,06 chez nous. Ce n'est donc pas propre à notre drizzle : c'est ce que fait un drizzle
  Bayer à l'échelle 1 sur 30 poses (ou 15 pour M8). Notre drizzle n'est pas plus bruité que celui de Siril (en moyenne
  0,05 mag de mieux à petite échelle, 0,08 à grande échelle) et va plus profond (+0,12 mag en moyenne).
- **La profondeur suit** : le drizzle de Siril perd 0,09 à 0,36 mag de profondeur face à son empilement classique
  (0,21 en moyenne), le nôtre 0,08 en moyenne (de −0,15 à +0,08).
- **Un point où nous sommes en retard** : le signal/bruit sur la cible. Le drizzle de Siril garde celui de son
  empilement classique (de −6 % à +13 %), le nôtre en perd 6 à 10 % sur les six cas. Notre drizzle reste donc 12 % en
  dessous de celui de Siril sur ce point (en moyenne), alors qu'il est devant à petite échelle : à regarder.
- **Les points d'une seule couleur (pixels chauds) sont aussi dans le drizzle de Siril**, et plus nombreux que chez
  nous sur Heart (76 par degré², contre 21 chez nous et 1 pour son empilement classique) et sur M8 (6 307 contre
  2 344) ; au même niveau sur Iris, M51 et la Rosette (16 contre 18 à 26). Les empilements classiques n'en ont presque
  plus (0 à 8 par degré² hors M8). Ce défaut est donc celui du drizzle Bayer, pas de notre implémentation.
- **Autre repère** : les étoiles faibles trouvées, la finesse et l'erreur de couleur bougent peu ou dans les deux sens
  (Heart : le drizzle de Siril a de moins bonnes couleurs que son classique, le nôtre de meilleures).

### Comment on l'a fait tourner

- **Même recette que « Siril classique »** (voir « La recette Siril »), mêmes fichiers, mêmes catalogues et réglages
  locaux, même post-traitement (astrométrie, extraction du fond de degré 2, PCC ou SPCC, étirement automatique). Seul
  change ce que le drizzle exige, d'après la documentation de Siril 1.4 et son script officiel
  `OSC_Preprocessing_BayerDrizzle.ssf` :
  - la calibration des poses garde les poses en mosaïque de Bayer (`-cfa -equalize_cfa`, sans `-debayer`) ;
  - le recalage devient `register pp_light -drizzle -scale=1.0 -pixfrac=1.0 -kernel=square -flat=<flat maître>`
    (la doc : « it is recommended to apply CFA drizzle with scale = 1.0 and pixfrac = 1.0 ») ;
  - l'empilement est la même ligne que pour l'empilement classique ;
  - Seestar (sans images de calibration) : `calibrate light` sans dématriçage, `register pp_light -2pass`, puis
    `seqapplyreg pp_light -drizzle -scale=1.0 -pixfrac=1.0 -kernel=square -filter-round=2.5k` (même filtre de rondeur
    que le script classique). Pas de flat pour ce cas.
- Aucun autre réglage de Siril n'a été touché. Calcul sur le processeur (Siril n'utilise pas la carte graphique ;
  nos mesures avec `CUDA_VISIBLE_DEVICES=-1`). Les résultats sont dans `siril_express_drizzle/` de chaque cas
  (`pre.ssf`, `post.ssf`, journaux, piles FITS, images finales, `siril.json`) ; les résultats de Siril classique
  (`siril_express/`) n'ont pas été touchés. Poses gardées : toutes (30 ; 15 pour M8).
- **Durée** (préparation entière de Siril, 8 cœurs) : drizzle contre classique, Heart 81 s / 122 s, Iris 93 s / 170 s,
  M51 125 s / 143 s, M8 248 s / 212 s, Seestar 19 s / 15 s, Rosette 96 s / 80 s.
- **Les mesures** sont celles du banc (mêmes fonctions que le banc croisé : `tools/bench/siril/measure_drizzle.py`
  appelle `crossed.py` et `measure.py`), sur le même ciel pour les quatre piles : le ciel couvert par toutes, la même
  liste d'étoiles Gaia, la même cible, des boîtes de même taille. Notre drizzle et notre empilement classique sont lus
  dans les résultats existants du banc croisé : le moteur n'a pas été relancé. Le ciel commun est un peu plus grand
  que celui du banc croisé (quatre piles seulement), donc quelques valeurs diffèrent de la 2e décimale des tableaux
  précédents. Les scripts (`run_siril_drizzle.py`, `measure_drizzle.py`, `report_drizzle.py`) sont copiés dans chaque
  dossier `siril_express_drizzle/` ; les mesures brutes sont dans `<cas>/drizzle_siril_express.json`.

### Les mesures, une table par mesure

Les deux dernières colonnes sont les écarts qui répondent à la question : drizzle moins classique, chez Siril puis
chez nous. Pour le bruit et la profondeur, négatif = le drizzle est moins bon.

**Profondeur : magnitude de l'étoile détectée à 5σ (plus haut = plus profond)**

| Cas | Siril classique | Siril drizzle | Notre drizzle | Notre classique | Siril : drizzle − classique | Nous : drizzle − classique |
|---|---:|---:|---:|---:|---:|---:|
| Heart | 16,33 | 16,00 | 16,23 | 16,37 | −0,33 | −0,13 |
| Iris | 17,42 | 17,15 | 17,27 | 17,42 | −0,27 | −0,15 |
| M51 | 18,29 | 18,20 | 18,21 | 18,20 | −0,09 | +0,01 |
| M8 | 17,75 | 17,65 | 17,86 | 17,78 | −0,09 | +0,08 |
| Seestar | 16,00 | 15,87 | 15,90 | 16,03 | −0,14 | −0,13 |
| Rosette | 17,32 | 16,96 | 17,05 | 17,18 | −0,36 | −0,13 |

**Bruit du fond, petite échelle (mag/arcsec² ; plus haut = moins de bruit)**

| Cas | Siril classique | Siril drizzle | Notre drizzle | Notre classique | Siril : drizzle − classique | Nous : drizzle − classique |
|---|---:|---:|---:|---:|---:|---:|
| Heart | 24,60 | 24,25 | 24,48 | 24,61 | −0,36 | −0,13 |
| Iris | 23,85 | 23,69 | 23,68 | 23,86 | −0,16 | −0,17 |
| M51 | 24,65 | 24,59 | 24,54 | 24,64 | −0,06 | −0,11 |
| M8 | 24,08 | 23,97 | 24,04 | 24,09 | −0,11 | −0,04 |
| Seestar | 22,63 | 22,48 | 22,52 | 22,64 | −0,14 | −0,12 |
| Rosette | 23,63 | 23,43 | 23,47 | 23,64 | −0,20 | −0,17 |

**Bruit du fond, grande échelle (mag/arcsec² ; plus haut = moins de bruit)**

| Cas | Siril classique | Siril drizzle | Notre drizzle | Notre classique | Siril : drizzle − classique | Nous : drizzle − classique |
|---|---:|---:|---:|---:|---:|---:|
| Heart | 25,29 | 25,00 | 25,24 | 25,29 | −0,30 | −0,05 |
| Iris | 25,22 | 25,07 | 25,10 | 25,21 | −0,14 | −0,11 |
| M51 | 25,91 | 25,87 | 25,88 | 25,86 | −0,04 | +0,01 |
| M8 | 24,39 | 24,29 | 24,37 | 24,39 | −0,10 | −0,02 |
| Seestar | 23,80 | 23,61 | 23,65 | 23,73 | −0,19 | −0,08 |
| Rosette | 24,82 | 24,65 | 24,74 | 24,82 | −0,17 | −0,08 |

**Signal/bruit sur la cible**

| Cas | Siril classique | Siril drizzle | Notre drizzle | Notre classique | Siril : drizzle − classique | Nous : drizzle − classique |
|---|---:|---:|---:|---:|---:|---:|
| Heart | 5,03 | 5,13 | 4,45 | 4,84 | +2 % | −8 % |
| Iris | 2,98 | 2,85 | 2,36 | 2,56 | −5 % | −8 % |
| M51 | 4,65 | 4,39 | 4,05 | 4,50 | −6 % | −10 % |
| M8 | 49,70 | 48,93 | 46,55 | 51,31 | −2 % | −9 % |
| Seestar | 19,84 | 22,35 | 20,43 | 21,71 | +13 % | −6 % |
| Rosette | 4,89 | 4,69 | 4,17 | 4,62 | −4 % | −10 % |

**Étoiles faibles trouvées à 5σ (les mêmes étoiles Gaia pour les quatre)**

| Cas | Siril classique | Siril drizzle | Notre drizzle | Notre classique | Siril : drizzle − classique | Nous : drizzle − classique |
|---|---:|---:|---:|---:|---:|---:|
| Heart | 22634 | 22126 | 22506 | 22837 | −2 % | −1 % |
| Iris | 4776 | 4519 | 4581 | 4807 | −5 % | −5 % |
| M51 | 1284 | 1241 | 1246 | 1233 | −3 % | +1 % |
| M8 | 27408 | 27563 | 27893 | 27367 | +1 % | +2 % |
| Seestar | 620 | 621 | 622 | 640 | +0 % | −3 % |
| Rosette | 5777 | 5471 | 5567 | 5704 | −5 % | −2 % |

**Finesse des étoiles : largeur à mi-hauteur au centre (″ ; plus bas = plus fines)**

| Cas | Siril classique | Siril drizzle | Notre drizzle | Notre classique | Siril : drizzle − classique | Nous : drizzle − classique |
|---|---:|---:|---:|---:|---:|---:|
| Heart | 13,10 | 12,34 | 12,07 | 13,01 | −6 % | −7 % |
| Iris | 6,12 | 6,55 | 6,15 | 6,08 | +7 % | +1 % |
| M51 | 6,88 | 7,09 | 6,95 | 6,86 | +3 % | +1 % |
| M8 | 5,63 | 5,69 | 5,33 | 5,58 | +1 % | −4 % |
| Seestar | 5,84 | 6,47 | 5,94 | 5,83 | +11 % | +2 % |
| Rosette | 5,81 | 5,99 | 5,96 | 5,90 | +3 % | +1 % |

**Points d'une seule couleur (pixels chauds restants, par degré²)**

| Cas | Siril classique | Siril drizzle | Notre drizzle | Notre classique | Siril : drizzle − classique | Nous : drizzle − classique |
|---|---:|---:|---:|---:|---:|---:|
| Heart | 1 | 76 | 21 | 1 | +75 | +21 |
| Iris | 3 | 16 | 26 | 6 | +13 | +19 |
| M51 | 4 | 16 | 18 | 5 | +12 | +12 |
| M8 | 1418 | 6307 | 2344 | 1196 | +4890 | +1148 |
| Seestar | 0 | 7 | 5 | 5 | +7 | +0 |
| Rosette | 6 | 16 | 22 | 8 | +9 | +14 |

**Couleurs des étoiles : erreur bleu−rouge face à Gaia (plus bas = plus fidèle)**

| Cas | Siril classique | Siril drizzle | Notre drizzle | Notre classique | Siril : drizzle − classique | Nous : drizzle − classique |
|---|---:|---:|---:|---:|---:|---:|
| Heart | 0,217 | 0,263 | 0,093 | 0,238 | +0,045 | −0,145 |
| Iris | 0,058 | 0,042 | 0,045 | 0,089 | −0,016 | −0,044 |
| M51 | 0,021 | 0,024 | 0,025 | 0,024 | +0,002 | +0,001 |
| M8 | 0,073 | 0,058 | 0,054 | 0,085 | −0,015 | −0,031 |
| Seestar | 0,158 | 0,147 | 0,133 | 0,090 | −0,011 | +0,042 |
| Rosette | 0,055 | 0,056 | 0,060 | 0,062 | +0,000 | −0,002 |

**Couleurs des étoiles : erreur vert−rouge face à Gaia (plus bas = plus fidèle)**

| Cas | Siril classique | Siril drizzle | Notre drizzle | Notre classique | Siril : drizzle − classique | Nous : drizzle − classique |
|---|---:|---:|---:|---:|---:|---:|
| Heart | 0,326 | 0,579 | 0,188 | 0,329 | +0,252 | −0,141 |
| Iris | 0,069 | 0,067 | 0,058 | 0,072 | −0,002 | −0,014 |
| M51 | 0,047 | 0,041 | 0,035 | 0,038 | −0,006 | −0,003 |
| M8 | 0,128 | 0,076 | 0,053 | 0,282 | −0,052 | −0,229 |
| Seestar | 0,262 | 0,243 | 0,166 | 0,165 | −0,019 | +0,001 |
| Rosette | 0,108 | 0,114 | 0,202 | 0,174 | +0,005 | +0,028 |

**Planéité du fond après retrait du dégradé (mag/arcsec² ; plus haut = plus plat)**

| Cas | Siril classique | Siril drizzle | Notre drizzle | Notre classique | Siril : drizzle − classique | Nous : drizzle − classique |
|---|---:|---:|---:|---:|---:|---:|
| Heart | 24,58 | 24,25 | 24,58 | 23,68 | −0,33 | +0,90 |
| Iris | 23,26 | 23,15 | 23,37 | 23,54 | −0,11 | −0,18 |
| M51 | 22,99 | 19,92 | 23,22 | 23,23 | −3,07 | −0,01 |
| M8 | 23,18 | 23,10 | 23,20 | 22,90 | −0,07 | +0,30 |
| Seestar | 21,29 | 21,37 | 21,01 | 21,34 | +0,08 | −0,34 |
| Rosette | 21,94 | 21,79 | 21,23 | 22,12 | −0,15 | −0,89 |

**Taux de rejet de l'empilement (% des échantillons écartés)**

| Cas | Siril classique | Siril drizzle | Notre drizzle | Notre classique | Siril : drizzle − classique | Nous : drizzle − classique |
|---|---:|---:|---:|---:|---:|---:|
| Heart | 0,941 | 0,990 | 0,014 | 1,097 | +0,049 | −1,083 |
| Iris | 0,892 | 0,788 | 0,044 | 1,347 | −0,104 | −1,303 |
| M51 | 0,804 | 0,596 | 0,024 | 0,852 | −0,207 | −0,828 |
| M8 | 1,222 | 1,177 | 0,014 | 1,215 | −0,045 | −1,201 |
| Seestar | 0,919 | 0,630 | 0,001 | 0,959 | −0,289 | −0,958 |
| Rosette | 0,910 | 0,841 | 0,039 | 0,545 | −0,069 | −0,506 |

À noter :
- **Taux de rejet** : les chiffres de notre drizzle (0,001 à 0,044 %) et ceux des deux empilements classiques ne sont
  pas comparables (notre drizzle compte les échantillons autrement) ; les deux Siril sont comparables entre eux : le drizzle
  de Siril rejette à peu près autant que son empilement classique (0,6 à 1,2 %).
- **M51, planéité du fond de Siril drizzle (19,92)** : la pile elle-même est normale (mêmes rapports de couleurs que
  Siril classique) ; c'est l'extraction du fond de degré 2 de Siril qui a mal ajusté le polynôme sur cette pile
  (dominante rose et verte sur l'image finale, rapport rouge/vert jusqu'à 1,5 dans un coin). Rien n'a été retouché.
- **M8, points d'une seule couleur** : les nombres sont grands pour les quatre piles (1 200 à 6 300 par degré²) parce
  que le champ est plein de nébulosité et d'étoiles ; seuls les écarts entre piles comptent.
- **Heart, couleurs des étoiles** : le drizzle de Siril donne l'erreur vert−rouge la plus forte (0,58, contre 0,33 en
  classique et 0,19 chez nous).
- Ces mesures portent sur l'Express (30 poses ; 15 pour M8) : elles disent ce que fait le drizzle à l'échelle 1 avec
  si peu de poses. Le drizzle de Siril n'a pas été mesuré en Base (100 poses).

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

Le flux « Classic stack » du moteur, réglages suggérés : calibration, alignement, empilement, astrométrie, couleur,
dégradé polynomial (degré 2), sortie simple. Le projet reçoit le même centre approximatif et la même focale que Siril.
L'état du moteur (profils d'appareil, cache des catalogues) est dans `F:/astro_essais/bench/_home` ; il lit d'abord
les catalogues locaux de Siril (`ASTRO_GPU_SIRIL_CATALOGUES`).

Pour le banc croisé, chaque cas a deux projets de plus, `v2x_express` et `v2x_base`, avec une chaîne par configuration
(OLD, S1 … V2-S6, nommées comme dans les tableaux) : le client les ouvre, et on peut y comparer les configurations
image par image. Seuls changent les interrupteurs de l'empilement v2 (avancés) ; tout le reste est à la valeur
suggérée. Les anciens projets `ours_express` et `ours_base` n'ont pas été touchés.

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

Ajoutées pour le banc croisé :
- **taux de rejet** : la part des échantillons écartés par l'empilement (chez Siril : sa ligne « Pixel rejection » de
  l'empilement des poses, moyenne des trois couleurs) ; Siril donne 0,1 à 0,5 % comme ordre de grandeur ;
- **poses gardées** : celles que l'empilement a vraiment utilisées ;
- **durée** : calibration + alignement + empilement, d'après les étapes du moteur (Siril : sa préparation entière).

## Rosette : les darks de la nuit d'Iris

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

Calibration + alignement + empilement, en minutes, et l'empilement seul entre parenthèses (Express). Siril : sa
préparation entière sur 8 cœurs (conversion comprise, ses durées du matin). Les nôtres sont mesurées pendant que
deux ou trois calculs du banc tournaient en même temps sur la machine (avec le client et Photoshop ouverts) : elles
sont pessimistes. La calibration est la même pour toutes les configurations d'un cas (calculée une fois).

<!-- crossed-durations:express -->
| Cas | Siril (préparation) | OLD | S1 | V2 | V2-C | S7 |
|---|---|---|---|---|---|---|
| heart | 2,0 | 2,3 (0,8) | 3,5 (1,9) | 2,9 (1,3) | 6,0 (4,4) | 3,6 (2,1) |
| iris | 2,8 | 2,3 (0,7) | 2,5 (0,9) | 2,4 (0,8) | 5,9 (4,3) | 3,6 (2,0) |
| m51 | 2,4 | 3,6 (1,3) | 4,0 (1,5) | 5,4 (2,9) | 6,0 (3,6) | 6,3 (3,9) |
| m8 | 3,5 | 6,8 (3,7) | 8,1 (5,0) | 7,5 (4,2) | 8,1 (4,8) | 7,9 (4,8) |
| seestar | 0,2 | 0,8 (0,1) | 0,9 (0,2) | 0,9 (0,2) | 5,3 (4,6) | 1,0 (0,3) |
| rosette | 1,3 | 2,5 (0,6) | 2,8 (0,9) | 3,1 (1,2) | 3,8 (1,9) | 3,5 (1,5) |
<!-- /crossed -->

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
5. **Passe finale** du 1er novembre : les six cas, Express et Base, avec le moteur corrigé ; Siril n'a pas changé.
   Ses tableaux sont remplacés par ceux du banc croisé.
6. **Retrait du correctif n° 3** (détail fin repris de la luminance : un bricolage, pas une méthode publiée), puis
   **l'empilement v2** (spec STACKING.md, six changements derrière des interrupteurs, commit 493df42) et **le banc
   croisé** (cette passe, 1er-2 novembre) : 19 configurations sur les six cas en Express, OLD, V2 et la configuration
   retenue en Base.
<!-- /history -->

## Limites du banc

- **Un passage par configuration.** La variation d'un passage à l'autre est mesurée par OLD-bis (un recalage à peine
  différent) : au plus 0,03 mag de profondeur et de bruit, 1,5 % d'étoiles faibles, 0,01 d'erreur de couleur. Les seuils
  du critère (0,04 mag, 3 %) sont au-dessus.
- **Le bruit dépend de l'échelle.** L'interpolation de la méthode Classique et de Siril (dématriçage, puis Lanczos)
  lisse le bruit d'un pixel à l'autre sans ajouter d'information ; notre drizzle garde chaque pixel brut. La grande
  échelle et les étoiles faibles trouvées sont les plus justes pour comparer l'information.
- **La « cible »** est la région étendue la plus brillante, prise automatiquement sur l'image de Siril.
- **La planéité** est faussée dans les champs pleins de nébulosité (Heart, Iris, M8, Rosette).
- **Les couleurs** sont jugées sur les étoiles, pas sur les nébuleuses ; la référence de blanc diffère d'un logiciel à
  l'autre.
- **Les points d'une seule couleur** ne veulent rien dire sur M8 (nébulosité rouge et étoiles faibles partout) : hors
  du critère.
- **Les poses écartées** : la sélection v2 peut écarter des poses ; Siril les garde toutes (sauf le script Seestar).
- **Les durées** sont prises sur une machine chargée (voir « Durées »).

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

Le banc croisé de l'empilement v2 (`tools/bench/crossed.py`) :

```
python tools/bench/crossed.py setup CAS NIVEAU                 # le projet v2x_<niveau> et une chaîne par configuration
python tools/bench/crossed.py phased CAS NIVEAU --device cuda:0  # jusqu'à l'astrométrie, cônes Gaia tirés du cache, puis le reste
python tools/bench/crossed.py setup CAS NIVEAU --configs V2-auto  # une fois V2 empilé (la méthode que la règle conseille)
python tools/bench/crossed.py measure CAS NIVEAU               # toutes les configurations et Siril sur le même ciel
python tools/bench/crossed.py sheet CAS NIVEAU --configs OLD V2  # planche côte à côte
python tools/bench/crossed.py report                           # les tableaux « crossed » de ce document
```

`phased` existe parce que chaque configuration a sa propre solution astrométrique, donc ses propres cellules de
catalogue : sans le découpage dans le cache, M8 attendait les services Gaia (VizieR muet, GAVO au-delà de 300 s,
l'ESA en erreur 500 le 1er novembre au soir).
