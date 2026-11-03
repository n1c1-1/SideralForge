# Chantier : le retrait du gradient

Branche `gradient`. Étapes 1 à 3 du plan : l'état de l'art, ce que fait notre code, la façon de mesurer.

## 1. L'état de l'art

| Outil | Méthode | Ce qu'il faut à l'utilisateur | Source |
|---|---|---|---|
| **Siril** | **Polynôme** (degré 1 à 4, au-delà instable) ou **RBF** (splines en plaque mince, avec un lissage) sur une grille d'échantillons. Un échantillon est rejeté s'il dépasse médiane + tolérance × MAD. Défauts : 20 échantillons par côté, tolérance 1,0, lissage 0,5. | retirer à la main les échantillons tombés sur les objets | [doc Siril](https://siril.readthedocs.io/en/latest/processing/background.html) |
| **Siril, version récente** | **Automatique, sans échantillons** : rejette par itérations tout ce qui ne ressemble pas au fond, puis ajuste un modèle multi-échelle. Défauts : échelle 5, douceur 1, protection des structures 0,05. | recadrer les bords d'abord | idem |
| **Siril + GraXpert** | Siril appelle le réseau de neurones de GraXpert par un script ; les modèles sont téléchargés par GraXpert. | installer GraXpert | [doc Siril](https://siril.readthedocs.io/en/latest/processing/graxpert.html) |
| **PixInsight ABE** | Polynôme automatique. Les utilisateurs le jugent le moins bon : il surcorrige les coins. | rien | forums |
| **PixInsight DBE** | Échantillons posés à la main, surface en splines. Le plus contrôlable, mais tout dépend du placement. | placer les échantillons | forums |
| **PixInsight GradientCorrection** (2024) | Automatique, à plusieurs échelles ; ses principes ne sont pas publiés. Résultats inégaux selon les images. | régler l'échelle et la protection | forums |
| **PixInsight MGC** | Le gradient est mesuré en comparant l'image à une **image de référence à grand champ** du même ciel. Seules les grandes échelles sont corrigées. La base de référence MARS (tout le ciel) est en préparation. | une image de référence, ou plus tard MARS | [tutoriel MGC](https://pixinsight.com/tutorials/multiscale-gradient-correction/), [MARS](https://pixinsight.com/doc/docs/MARS/MARS.html) |
| **GraXpert IA** | Un réseau de neurones prédit le fond, sans échantillons. Réglages : lissage (0 à 1), soustraction ou division. Ligne de commande : `GraXpert image.fits -cli -cmd background-extraction`. | rien | [GraXpert](https://github.com/Steffenhir/GraXpert) |
| **GraXpert classique** | RBF, splines ou krigeage sur des échantillons choisis. | placer les échantillons | idem |

**Les licences de GraXpert** :
- **le programme** est sous **GPL-3.0** ([License.md](https://github.com/Steffenhir/GraXpert/blob/main/License.md)) ;
- **les modèles de réseau de neurones** sont sous **CC BY-NC-SA 4.0**, d'après un enregistrement du dépôt officiel et un miroir. C'est donc un **usage non commercial seulement**. À confirmer sur les fichiers officiels.

**Ce que ça implique** :
- Embarquer les modèles dans notre logiciel le soumettrait à cette clause non commerciale. Un utilisateur qui vend ses tirages serait en dehors de la licence.
- Si l'on **appelle GraXpert installé par l'utilisateur**, comme le fait Siril, la licence reste entre l'utilisateur et GraXpert.
- **Ma recommandation : l'appeler comme un programme externe.**

**Remarques** :
- Il n'existe aucune comparaison chiffrée et publiée entre ces outils, seulement des avis d'utilisateurs.
- L'idée de MGC (s'appuyer sur une référence extérieure du ciel) est proche de notre fond « ancré » sur un relevé H-alpha.

## 2. Ce que fait notre code

**Le fond polynomial** (`background_poly`, workflows Empilement classique et Classique + débruitage) :
- Il suit la méthode habituelle de Siril et d'ABE : un polynôme ajusté sur des blocs du ciel, en rejetant par itérations les blocs trop au-dessus de la surface.
- Réglages : degré 2, 16 échantillons par côté, rejet à 2,5 σ. Un grand objet qui remplit le champ (M31) est laissé de côté en entier.
- **Généralisable : oui**, il ne contient pas de règle calée sur nos images.
- Il lui manque la RBF, la méthode automatique et les échantillons posés à la main.

**Le fond « physique »** (`background`, workflow Complet) est une **construction maison** :
1. un « flat des étoiles » (le flux mesuré des étoiles divisé par le flux prédit par Gaia) corrige l'image ;
2. le ciel est modélisé comme *relevé H-alpha × un gain*, plus *lumière des étoiles non résolues prédite par Gaia × un gain*, plus un polynôme. Seul le polynôme est retiré ;
3. le zéro du fond est extrapolé des comptages d'étoiles Gaia ;
4. une correction à grande échelle rapproche l'image du relevé, à la manière de MGC.

**Les réglages au cas par cas qu'il contient** :
- un plancher fixé sur les zones les plus sombres ;
- une précision du halo des étoiles mesurée autour d'Alnitak ;
- un rejet des objets fait sur le vert et le bleu quand l'amplitude H-alpha s'effondre (un champ d'émission) ;
- des seuils fixes (20 %, 0,2 centile…) ;
- dans le code, les commentaires citent Horsehead et Alnitak pour justifier ces règles.

**Généralisable : douteux.**
- Il suppose un relevé H-alpha utilisable, des étoiles Gaia bien mesurées et un appareil couleur.
- Ses règles ont été ajoutées image par image.
- **Aucune méthode publiée ne le décrit dans son ensemble.** Ses idées de départ, elles, sont publiées : la référence extérieure de MGC, et le flat des étoiles, une calibration photométrique classique.

## 3. La façon de mesurer

Les mesures portent sur les empilements du banc déjà calculés (Heart, Iris, M51, M8, Rosette, Seestar), sans rien réempiler.

1. **Un gradient injecté, la seule mesure qui donne une vérité.**
   - On ajoute à l'image un gradient connu (un plan incliné, un dôme, un coin éclairé par la Lune).
   - Chaque méthode est appliquée avec et sans cet ajout. L'écart entre les deux fonds trouvés doit être le gradient ajouté.
   - Mesure : le gradient retrouvé à 1 % près, ou pas.
2. **Un objet faible injecté.** On ajoute un nuage très large et très faible (le niveau de l'IFN). Mesure : la part qui survit au retrait. Une bonne méthode le garde.
3. **Sur les vraies images** :
   - la planéité du fond du ciel (l'écart entre les zones de ciel, sur un masque des objets fait indépendamment) ;
   - le bas des extensions des nébuleuses (Heart, Rosette) et du halo de M51.
4. **Ton œil, dans le GUI**, avec l'avant, l'après et le modèle retiré côte à côte.

**Méthodes comparées** :
- **Siril** (1.4.0-beta3 installé) : polynôme et RBF ;
- **GraXpert** : réseau de neurones et RBF. Il faut l'installer, ce que je ne ferai qu'avec ton accord ;
- **nos deux méthodes**.

PixInsight n'est pas installé : il reste comparé sur la documentation et les publications.

## 4. Les résultats du banc (3 novembre)

Six cas, cinq versions de chaque image, six méthodes avec leurs réglages par défaut. Outil : `tools/bench/gradient.py` ;
chiffres : `F:\astro_essais\bench\gradient\results.json` ; planches : `F:\astro_essais\bench\gradient\sheet_<cas>.jpg`
(l'image d'entrée, chaque résultat, chaque fond retiré).

Méthodes : Siril 1.4.5 (polynôme degré 2 et 4, RBF), Siril 1.5 en développement (automatique, sans échantillons),
GraXpert 3.1.0-RC2 (réseau de neurones), notre polynôme (degré 2).

| Méthode | Gradients injectés : écart au vrai (pente, dôme, coin éclairé) | Nuage faible gardé | Planéité du fond |
|---|---|---|---|
| Notre polynôme | **1 à 3 %**, **1 à 3 %**, 24 à 51 % | **76 à 104 %** | 3,6 à 8,5 % |
| Siril polynôme 2 | 5 à 17 %, 8 à 37 %, 27 à 57 % | 82 à 95 % | 3,7 à 9,6 % |
| Siril polynôme 4 | 7 à 71 %, 25 à 44 %, 9 à 50 % | 52 à 85 % | 3,3 à 9,2 % |
| Siril RBF | 9 à 21 %, 25 à 42 %, 11 à 47 % | 36 à 78 % | 2,0 à 8,3 % |
| Siril automatique (1.5) | **2 à 4 %** (M8 : 18 %), **5 à 10 %** (M8 : 27 %), **3 %** (M8 : 14 %) | 11 à 20 % (M8 : 53 %) | **1,6 à 3,5 %** (M8 : 6,4 %) |
| GraXpert IA | 7 à 27 %, 9 à 43 %, 7 à 37 % | 13 à 24 % (M8 : 76 %) | 1,6 à 9,1 % |

Ce que ça dit :
- **Deux familles, un même compromis.**
  - **Les méthodes rigides** (les polynômes) **gardent le signal faible et étendu**, mais ne suivent pas un gradient
    qui n'a pas une forme de polynôme : le coin éclairé par la Lune laisse 24 à 57 % d'erreur.
  - **Les méthodes souples** (Siril automatique, GraXpert, et la RBF dans une moindre mesure) **suivent les
    gradients compliqués**, mais **retirent aussi les grandes structures faibles**. Sur la Rosette, la planche montre
    que la nébuleuse elle-même passe en partie dans le fond retiré, avec GraXpert comme avec Siril automatique.
- **Notre polynôme est meilleur que celui de Siril** sur les gradients lisses (1 à 3 % contre 5 à 37 %). Siril écarte
  les échantillons plus brillants que la médiane de toute l'image : avec un gradient fort, il ignore tout le côté
  éclairé et extrapole depuis le côté sombre. Sa documentation conseille d'ailleurs de garder tous les échantillons
  dans ce cas. Réserve : la pente et le dôme injectés ont une forme de polynôme, ce qui avantage les polynômes. Le coin
  éclairé est la mesure honnête pour eux.
- **Siril automatique est la meilleure des méthodes souples** : il retrouve les trois gradients à quelques % près, mieux
  que GraXpert, et donne le fond le plus plat.
- **M8 (Voie lactée dense) met toutes les méthodes en difficulté.**
- **Une méthode qui ne regarde que l'image ne peut pas distinguer un nuage faible très étendu d'un gradient.** Seule une
  référence extérieure du ciel le peut : c'est l'idée de MGC et de MARS chez PixInsight, et celle de notre fond ancré
  sur le relevé H-alpha. C'est la piste de l'état de l'art pour garder les nébulosités faibles.
- **Notre fond « physique » n'est pas dans ce banc** : il demande toute la chaîne du workflow Complet. Ses résultats
  précédents sur Rosette et Iris restent visibles dans le GUI.

## 5. La méthode ancrée sur le relevé H-alpha (3 novembre, soir)

**Ce qu'elle fait** : la même surface lisse que notre polynôme, mais le relevé H-alpha professionnel du champ entre
dans l'ajustement comme une partie du vrai ciel (gain × relevé + surface). Seule la surface est retirée. C'est le
principe de MGC chez PixInsight (une image du même ciel, sans gradient, dit ce qui est du ciel), limité ici à ce que
voit le relevé : l'émission H-alpha. Méthode `anchored` de l'étape Gradient ; elle a besoin du relevé (étape
`surveys`) et donc d'Internet.

**Le banc** : les mêmes six cas, mais pris désormais **en couleurs de l'appareil**, comme la nouvelle étape les voit
(avant la calibration des couleurs). Toutes les méthodes ont été relancées sur cette entrée. Une mesure s'ajoute :
une émission faible ayant la forme que montre le relevé est injectée, et l'on mesure la part qui reste. C'est le cas
le plus favorable à la méthode ancrée (sa référence a exactement cette forme). Chiffres :
`F:\astro_essais\bench\gradient_anchored\results.json`, planches à côté.

| Méthode | Gradients injectés : écart au vrai (pente, dôme, coin) | Nuage neutre gardé | Émission du relevé gardée | Planéité |
|---|---|---|---|---|
| Notre polynôme | 0 à 2 %, 0 à 2 %, 23 à 52 % | 33 à 108 % | 65 à 94 % | 3,1 à 15 % |
| **Ancrée** | 0 à 2 %, 1 à 2 %, 33 à 52 % | 19 à 103 % | **96 à 101 %** (Iris : 135 %) | 3,0 à 12 % |
| Siril polynôme 2 | 5 à 15 %, 5 à 30 %, 26 à 56 % | 80 à 96 % | 85 à 103 % | 3,6 à 17 % |
| Siril automatique (1.5) | 1 à 5 % (M8 : 21 %), 5 à 13 % (M8 : 30 %), 3 à 4 % (M8 : 16 %) | 13 à 24 % (M8 : 64 %) | 26 à 100 % | 1,2 à 11 % |
| GraXpert IA | 4 à 14 %, 6 à 17 %, 5 à 12 % | 10 à 22 % (M8 : 75 %) | 88 à 106 % | 1,1 à 17 % |

Ce que ça dit :
- **L'ancrage garde l'émission que le relevé voit** : 96 à 101 % contre 65 à 94 % pour le polynôme seul. Sur Rosette,
  Siril automatique n'en garde que 26 % (la planche montre la nébuleuse creusée).
- **Il ne rend pas la surface plus souple** : le coin éclairé reste à 33 à 52 % d'erreur, comme avec le polynôme.
- **Il a un défaut** : sur Iris (peu d'H-alpha, beaucoup de poussière), le gain du relevé s'emballe et l'émission
  injectée ressort à 135 %. Le relevé y ressemble à la poussière, et l'ajustement lui attribue de la lumière qui n'est
  pas la sienne. Sur Rosette, le nuage neutre injecté (absent du relevé) est moins bien gardé (19 % contre 33 %).
- **M8 manque** : trop au sud pour NSNS, et le service du CDS qui sert le relevé de Finkbeiner ne répondait pas.
- **L'entrée en couleurs de l'appareil change les chiffres** par rapport au tableau du 3 novembre : le ciel très
  vert de l'appareil pèse dans la planéité, pour toutes les méthodes. Sur Rosette, notre polynôme garde 33 % du nuage
  neutre (93 % avant) ; c'est à comprendre.

**Décision (3 novembre, 17h20) : la méthode ancrée est abandonnée**, retirée du moteur. Elle ne rend pas la surface
plus souple, et seul le relevé H-alpha est à notre portée comme référence. Le chantier continue avec la réécriture de
la méthode automatique de Siril (étape Gradient) et un retrait du gradient image par image avant l'empilement.

## 6. La méthode automatique de Siril, réécrite (3 novembre, soir)

**Ce qu'elle est.** La méthode « automatique, sans échantillons » de Siril 1.5 (`subsky -auto`), réécrite dans notre
moteur (`astro_gpu/stages/gradient_auto.py`, méthode `automatic` de l'étape Gradient). Sa marche vient de la
documentation de Siril ; les valeurs que la documentation ne donne pas viennent de leur code source (version du
2 novembre) : rejet au-dessus de médiane + 2 σ et au-dessous de médiane − 4 σ, 20 tours au plus, flou de rayon 5 % du
petit côté de l'image réduite 4 fois, bouchage des trous en 10 passes, masque de protection des structures. Notre code
est écrit à partir de cette description ; il ne traduit pas le leur.

**La fidélité, mesurée.**
- Les fonctions du code source de Siril, compilées à part, donnent **exactement** nos résultats (écart 0,00004).
- Mais **le programme Siril distribué ne fait pas exactement ce que dit son code source.** Deux différences trouvées
  et reproduites :
  1. aux bords, le programme continue l'image selon sa pente (le code source répète le dernier pixel, ce qui fait
     s'affaisser le fond aux bords puis dériver tout le modèle tour après tour) ;
  2. les statistiques du rejet se comportent comme si elles portaient sur toute l'image, pas sur les seuls pixels
     gardés.
- Avec ces deux corrections, l'écart entre notre fond et celui du programme Siril est de **1 à 10 % de l'amplitude du
  fond** selon les cas (le plus grand sur M8 et le Seestar). Une troisième différence reste inexpliquée ; le nombre de
  passes, de tours, la règle d'arrêt et la protection ont été testés sans la trouver.

**Le banc** (mêmes six cas, en couleurs de l'appareil ; Siril : `F:\astro_essais\bench\gradient_auto`, notre version :
`F:\astro_essais\bench\gradient_auto_v2`).

| Cas | Méthode | Pente | Dôme | Coin | Nuage gardé | Planéité |
|---|---|---|---|---|---|---|
| Heart | Siril | 1,3 | 4,7 | **2,5** | 15 | 3,6 |
| | nôtre | 3,5 | 6,6 | 9,7 | 20 | 3,6 |
| Iris | Siril | 2,9 | 5,6 | 2,9 | 14 | 2,7 |
| | nôtre | 2,5 | 2,4 | 2,2 | 18 | 2,8 |
| M51 | Siril | 2,6 | 5,7 | 3,5 | 16 | 1,2 |
| | nôtre | 2,2 | 2,6 | 2,9 | 20 | 1,5 |
| M8 | Siril | 21 | 30 | 16 | 64 | 11 |
| | nôtre | **4,1** | **10** | **4,2** | 60 | 12 |
| Rosette | Siril | 2,5 | 5,3 | 2,7 | 24 | 5,0 |
| | nôtre | 2,1 | 2,2 | 2,3 | 27 | 5,2 |
| Seestar | Siril | 5,3 | 13 | 3,8 | 13 | 1,5 |
| | nôtre | 2,8 | 9,2 | 2,8 | 16 | 1,5 |

(écarts au gradient injecté en %, nuage gardé en %, planéité en % du niveau du ciel)

Ce que ça dit :
- **Notre version fait jeu égal ou mieux que Siril** sur cinq cas, nettement mieux sur M8 ; elle garde un peu plus du
  nuage faible.
- **Une exception : le coin éclairé de Heart** (9,7 % contre 2,5 %). C'est la trace de la différence inexpliquée.
- Comme Siril, elle **retire la plus grande partie d'un nuage faible et très étendu** (15 à 27 % gardés, M8 à part) :
  c'est le prix de la souplesse. Le polynôme, lui, le garde (33 à 108 %) mais rate le coin éclairé.
