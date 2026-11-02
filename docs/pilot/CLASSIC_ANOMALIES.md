# Les deux anomalies de notre empilement classique (2 novembre)

Deux choses bizarres sont apparues au banc de ce matin sur notre empilement classique (dématriçage RCD, comme Siril).
Chacune a été mesurée étape par étape sur les mêmes données, contre Siril classique et contre notre drizzle. Les
mesures sont dans `F:/astro_essais/bench/<cas>/anomalies_depth.json` et `anomalies_colour.json`. L'outil est
`tools/bench/classic_anomalies.py`.

## Anomalie 1 : l'équilibrage avant le dématriçage « coûte » 0,57 mag de profondeur sur Heart

**Ce qu'on voit.** Avec l'équilibrage des couleurs avant le dématriçage (le *dématriçage* reconstruit les trois
couleurs à chaque pixel à partir de la mosaïque du capteur), la profondeur du banc tombe de 16,38 à 15,81 mag sur Heart.
Sur les autres cas, l'écart va de −0,08 à +0,08 mag.

**D'où ça vient : surtout de la mesure elle-même (0,45 mag sur les 0,57).** La profondeur du banc compare le flux des
étoiles au bruit mesuré dans de petits disques posés sur le ciel vide. Le rayon d'un disque vaut une largeur d'étoile
(*FWHM*, la largeur à mi-hauteur), et le disque n'est fait que de pixels entiers. Or la largeur des étoiles de Heart
passe de 2,17 à 2,24 pixels, juste au-dessus de 2,236 (√5). À ce seuil, le disque passe d'un coup de 13 à 21 pixels.
Il ramasse alors 56 % de bruit en plus, sans gagner de signal : la profondeur perd 0,45 mag.
- Avec le même disque de 13 pixels pour les deux images, l'écart n'est plus que de 0,11 mag.
- Avec une mesure continue, l'écart est de 0,14 mag (16,09 contre 15,95). Cette mesure prend des disques exacts au
  pixel près, de même rayon en secondes d'arc pour toutes les images, et corrige la part du flux qui sort du disque :
  c'est la méthode des grands relevés.
- Le même effet de seuil fausse d'autres lignes du banc. Sur M8, le drizzle a un disque de 21 pixels et Siril de 25 :
  le banc donne notre drizzle devant de 0,12 mag, la mesure continue le donne derrière de 0,02. Sur la Rosette, le banc
  donne notre drizzle 0,24 mag derrière Siril, et la mesure continue 0,04 mag derrière.
- **À corriger dans l'outil du banc** (`tools/bench/measure.py`, que je n'avais pas le droit de modifier) : des disques
  exacts avec la correction du flux, comme dans `classic_anomalies.py`.

**Le reste (environ 0,1 mag) n'est pas une perte : c'est de la lumière verte recopiée.** Sans équilibrage, RCD fait
passer dans le bleu et le rouge des étoiles une partie de la lumière du vert. On l'a mesuré sur une pose de Heart, en
comparant le flux de 3 000 étoiles à la somme exacte des pixels de chaque couleur :

| Pose de Heart | Rouge | Vert | Bleu |
|---|---:|---:|---:|
| Sans équilibrage (comme Siril) | +4 % | −6 % | **+36 %** |
| Avec équilibrage | −6 % | −3,5 % | −5 % |

- La luminance du banc est la moyenne des trois couleurs. Elle compte ce vert recopié comme du signal bleu : sans
  équilibrage, les étoiles paraissent plus brillantes face au bruit.
- Couleur par couleur, sur l'empilement : en vert seul, les deux versions sont à égalité (15,91 contre 15,89 mag). En
  bleu, la version sans équilibrage paraît plus profonde de 0,30 mag, en rouge de 0,09 mag.
- Le bruit du ciel en bleu est même plus faible avec l'équilibrage : −30 % au pixel, −14 % sur 8 pixels.
- Les étoiles 1 à 3 % plus larges viennent du bleu (Heart : 14,7″ contre 13,7″ ; Iris : 7,2″ contre 6,6″). Sans
  équilibrage, le bleu prend la forme plus fine du vert. Avec, il est fait davantage de ses propres pixels, qui ne
  sont qu'un sur quatre.

**Ce qu'on a vérifié et qui n'y est pour rien.** Le rejet (1,06 % contre 1,10 % des valeurs), la normalisation
(identique), l'écrêtage (aucun), l'étape couleur (la profondeur est mesurée avant elle), les étoiles de mesure (les
mêmes étoiles Gaia, le même point zéro 25,01). Notre RCD est identique à celui de Siril : écart de 4 millionièmes de la
dynamique.

**Les six cas** (avec équilibrage moins sans ; en mag, négatif = l'équilibrage paraît moins profond) :

| | Heart | Iris | M51 | M8 | Seestar | Rosette |
|---|---:|---:|---:|---:|---:|---:|
| Bleu des étoiles en trop sans équilibrage (par rapport au vert) | +45 % | +16 % | +2 % | +12 % | +8 % | +18 % |
| Écart de profondeur, mesure du banc | −0,57 | −0,07 | +0,08 | −0,04 | −0,04 | −0,08 |
| Écart de profondeur, mesure continue | −0,14 | −0,01 | +0,08 | −0,05 | −0,05 | −0,05 |

L'écart suit la lumière recopiée : grand sur Heart (ciel bleu très faible, 0,4 fois le vert), nul sur M51.

**À trancher.** Il n'y a pas de défaut dans notre code, donc rien n'a été changé. Le choix est une question de méthode :
- **Avec équilibrage** (réglage actuel) : c'est l'ordre de dcraw, RawTherapee et darktable, qui équilibrent les
  couleurs avant de dématricer. Le flux de chaque couleur est juste à quelques % près, et cela corrige les couleurs des
  étoiles de Heart (erreur B−R 0,153 au lieu de 0,238). En revanche, les étoiles sont un peu plus larges en bleu, et la
  luminance perd jusqu'à 0,14 mag (Heart).
- **Sans équilibrage** : c'est ce que fait Siril 1.4. Le bleu et le rouge des étoiles sont gonflés par le vert, de +2 à
  +45 % selon le ciel, mais la luminance paraît un peu plus profonde.

## Anomalie 2 : sans équilibrage, Heart sort orange et granuleux

**Ce qu'on voit.** Notre classique sans équilibrage a le bruit de Siril classique, à 0,02 mag près à toutes les
échelles et dans chaque couleur. Pourtant, sur la planche, il sort orange, granuleux et comme « posterisé ».

**Où les deux divergent : à notre étape couleur, pas avant ni après.**
1. **L'empilement** est le même que celui de Siril (bruit par couleur R/V/B à 1 et 8 pixels : 23,86/24,03/23,84 et
   24,54/24,79/24,87 chez nous, contre 23,85/24,02/23,83 et 24,54/24,80/24,89 chez Siril).
2. **L'étape couleur** reconstruit les courbes de sensibilité du capteur à partir des couleurs de milliers d'étoiles
   Gaia, puis en tire une *matrice* (un mélange des trois couleurs du capteur). Sur cet empilement, les couleurs des
   étoiles sont écrasées vers le gris, deux fois moins contrastées qu'avec notre drizzle, et teintées par le vert
   recopié. Les courbes trouvées ne sont donc pas physiques (le rouge est centré à 590 nm). L'étape a alors pris les
   courbes d'un capteur « type » (un A7S non modifié, alors que celui de Heart est défiltré), **sans vérifier qu'elles
   prédisent les étoiles du champ**. Elles se trompent de 17,9 %, alors que la règle pour un profil d'appareil est 5 %.
3. **La matrice obtenue** mélange les couleurs avec de gros coefficients de signes opposés (rouge de sortie = 4,0 R −
   4,0 V + 2,0 B). Elle multiplie le bruit du rouge par 2,4 par rapport à un simple facteur par couleur. Elle rend aussi
   l'hydrogène alpha orange (1 : 0,42 : 0,11) au lieu de rouge-magenta (1 : 0 : 0,30, comme le rend notre drizzle).
4. **Sur la planche**, dans la zone faible, le bruit rouge vaut 4,2 fois celui de Siril, et 34 % des pixels tombent
   au noir de la planche, contre 15 % chez Siril. C'est cela, le grain « posterisé ». La planche, le dégradé et la
   sortie n'y sont pour rien : la même image calibrée avec un facteur par couleur ressemble à Siril (voir l'image).

`shots/2026-11-02/anomalies/heart_couleur_avant.png` (avant) et `heart_couleur_apres.png` (après) montrent, à 100 %
et avec le même étirement que la planche du banc : Siril, notre drizzle, notre classique avec puis sans équilibrage,
et le même calibré avec un facteur par couleur.

**Ce qui a été corrigé (étape couleur, version 5).** La règle qui protège déjà les profils d'appareil s'applique
maintenant aussi aux courbes « type » : il faut prédire les étoiles du champ à 5 % près. Si ni les courbes ajustées, ni
un profil, ni les courbes type ne passent, l'étape ne fait plus de matrice. Elle prend un facteur par couleur, tiré de
la balance des blancs des étoiles : c'est la méthode des calibrations photométriques de Siril (PCC, SPCC) et de
PixInsight (SPCC). Cela ne touche que trois images du banc : Heart classique avec et sans équilibrage, et la Rosette
classique sans équilibrage. Les autres images du banc gardent leur matrice.

| Heart, avant → après | Siril classique | Notre classique | Notre classique sans équilibrage |
|---|---:|---:|---:|
| Bruit de la zone faible R/V/B (× la luminance de Siril) | 1,28/1,02/1,18 | 2,06/1,37/2,12 → 1,28/1,16/1,15 | 4,20/2,08/2,12 → 1,22/1,30/1,14 |
| Pixels au noir sur la planche | 15 % | 27 % → 16 % | 34 % → 17 % |
| Planéité du fond (mag/arcsec²) | 24,59 | 24,37 → 24,50 | 23,69 → 24,59 |
| Erreur de couleur B−R | 0,217 | 0,153 → 0,194 | 0,238 → 0,223 |
| Erreur de couleur V−R | 0,326 | 0,283 → 0,363 | 0,329 → 0,489 |

Sur la Rosette sans équilibrage, c'était un cas limite : les courbes type s'y trompaient de 5,5 %, juste au-dessus de la
règle, et la matrice y était sage. La correction y fait un peu moins bien sur les couleurs des étoiles (B−R 0,056 →
0,060, V−R 0,152 → 0,181, Siril 0,053 et 0,102) et un peu mieux sur la planéité (22,10 → 22,19). Les mesures d'après
sont dans `F:/astro_essais/bench/<cas>/today_express_colour5.json`. Les tableaux de BENCH.md restent ceux d'avant.

Le prix à payer : **les couleurs des étoiles du classique ne sont plus « redressées »** par une matrice, elles restent
aussi écrasées que chez Siril. Les erreurs de couleur du banc sont plus fortes pour le classique avec équilibrage. Une
partie de cette hausse vient du fait que le banc mesure maintenant deux fois plus d'étoiles, plus faibles, puisque le
bruit a baissé (9 600 contre 4 700 à 6 200). Sur les mêmes 3 110 étoiles, l'erreur B−R passe de 0,234 à 0,254 avec
équilibrage, et de 0,390 à 0,311 sans.

**À trancher.**
- **Garder la correction** (mon conseil) : quand aucune courbe ne décrit le capteur, on fait comme Siril, sans grain
  ni teinte inventés. Pour revenir en arrière, il suffit d'une ligne de `astro_gpu/stages/color.py`.
- **À savoir, hors de ce correctif** : même quand les courbes sont bonnes, notre matrice coûte du bruit de couleur
  face à un facteur par couleur. Elle le multiplie jusqu'à 4,5 fois selon le cas et la couleur (notre drizzle de
  Heart : ×1,4 ; Seestar et Rosette : jusqu'à ×4). C'est le prix des couleurs « vraies » de notre étape couleur, qui va
  au-delà de Siril. Une option « couleurs comme Siril (un facteur par couleur) » serait l'équivalent exact de
  Siril 1.4. C'est une décision de produit, pas un défaut.
