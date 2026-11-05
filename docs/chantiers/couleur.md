# Chantier : la calibration des couleurs

Branche `couleur`. Ouvert le 3 novembre au soir, après les couleurs mauves de la Rosette (session réduite) dans le
Post-traitement.

## 1. L'état de l'art

| Outil | Méthode | Ce qu'il faut | Source |
|---|---|---|---|
| **Siril SPCC** | Pour chaque étoile du champ, le flux attendu dans chaque canal est calculé à partir de son spectre Gaia DR3 et des **courbes de sensibilité connues** du capteur et des filtres. Correction : **un facteur par canal** (une balance des blancs), pas de matrice. | le modèle de capteur et de filtres, dans une base communautaire (GPL-3) | [doc Siril](https://siril.readthedocs.io/en/latest/processing/color-calibration/spcc.html), [base](https://gitlab.com/free-astro/siril-spcc-database) |
| **Siril PCC** (plus ancien) | Même principe, sans les spectres : couleurs photométriques des étoiles. Moins précis. | rien | idem |
| **PixInsight SPCC** | Présenté comme une **balance des blancs** fondée sur Gaia, par rapport à une référence de blanc. La documentation ne parle pas de matrice ; un utilisateur affirme qu'il n'y en a pas (source informelle, à vérifier). | les courbes du capteur et des filtres | [doc PixInsight](https://pixinsight.com/doc/docs/SPCC/SPCC.html) |
| **Photographie classique** (RawTherapee, darktable, Adobe) | Balance des blancs puis **matrice de correction des couleurs propre à chaque modèle d'appareil**, mesurée par le fabricant ou par Adobe (DNG). Valable pour un appareil **non modifié** seulement. | le modèle d'appareil | matrices Adobe DNG, reprises par libraw |

Ce que ça dit :
- **Le plancher (Siril, PixInsight) est une balance des blancs, sans matrice.** Une balance des blancs ne mélange pas les
  canaux : elle ne peut pas créer de mauve à partir de rouge. Elle laisse aussi la teinte telle que les filtres du
  capteur la donnent : une émission H-alpha vue un peu par les pixels bleus reste un peu bleue.
- **Une vraie matrice demande les courbes du capteur**, connues (base SPCC, matrices des fabricants), pas devinées.
- **La base SPCC de Siril couvre 2 de nos 4 appareils** : la caméra ZWO ASI2600MC (capteur Sony IMX571, M8) et le
  Seestar S50. Elle n'a ni le Sony A7S (Heart, Iris, Rosette ; défiltré pour Heart au moins), ni le Canon 6D (M51).
  Sa licence est la GPL-3 : comme pour GraXpert, on la lirait là où l'utilisateur l'a installée, sans l'embarquer.

## 2. Ce que fait notre étape Couleur

Elle n'est pas dans l'état de l'art : c'est une **construction maison**.
1. Elle mesure quelques milliers d'étoiles dont Gaia donne le spectre complet.
2. Elle **reconstruit les courbes de sensibilité de l'appareil** à partir de ces étoiles (« response_source : fit »).
3. Elle en déduit une **matrice 3 × 3** caméra → sRGB, ajustée sur les étoiles, sur des spectres de synthèse (corps noirs
   rougis), et **ancrée sur un spectre H-II type** (poids 60), pour que l'hydrogène ressorte avec sa vraie couleur.
4. Elle garde les courbes trouvées comme « profil » de l'appareil, réutilisé quand un champ ne suffit pas.
5. Une option « un facteur par couleur » fait la balance des blancs seule, comme SPCC.

## 3. Le diagnostic (3 novembre)

**Les couleurs de la Rosette (session réduite) :** la nébuleuse sort en rouge : vert : bleu = 1 : 0,44 : 0,65 (plus de
bleu que de vert : mauve). Le spectre H-II type a pour vraie couleur 1 : −0,01 : 0,26 (sRGB linéaire).

**Ce n'est pas le gradient** : la matrice fautive est identique, au chiffre près, à celle calculée le matin même, avant
le déplacement du gradient.

**La matrice est mal déterminée sur ce cas** : sur des données presque identiques, l'étape a donné trois matrices très
différentes (le 1er novembre, ce matin, ce soir après le gradient automatique), qui collent toutes aux étoiles à 1,7–1,9 %
près. Sur les 11 projets du banc, les deux calculs de couleur de chaque projet (après les deux gradients) concordent à
0,25 près au pire, sauf la Rosette réduite (2,12).

**L'ancrage H-II ne garantit pas la teinte réelle.** Partout, la matrice rend le spectre H-II type juste
(1 : −0,01 : 0,29), mais **calculé avec les courbes que l'étape a reconstruites**. Si ces courbes se trompent sur ce que
l'appareil enregistre réellement à 656 nm (les étoiles, aux spectres larges, ne le disent pas), la nébuleuse réelle sort
d'une autre couleur. C'est l'explication la plus probable du mauve ; elle se vérifie (section 4).

**Une part du défaut vient du déplacement du gradient** : la sortie simple prend désormais comme niveau du ciel la
médiane de chaque canal sur tout le champ. Sur la Rosette, la nébuleuse en fait partie : le fond n'est plus neutre
(rouge −3,5, vert −0,4, bleu −1,8). Ça teinte le fond, pas la nébuleuse. Correction prévue : prendre le niveau du ciel
mesuré par le bloc Gradient, qui ignore les objets.

## 4. La façon de mesurer

1. **La vérité terrain des courbes.** Pour les deux appareils de la base SPCC (ASI2600MC, Seestar S50), on connaît les
   vraies courbes : on compare ce que nos courbes reconstruites prédisent pour l'H-alpha, l'[OIII] et les étoiles à ce
   que prédisent les vraies. C'est la seule mesure qui dise si notre reconstruction est juste.
2. **La stabilité.** Recalculer la couleur sur des tirages d'étoiles différents (la moitié des étoiles, au hasard) : de
   combien bougent la couleur rendue de l'H-alpha, de l'[OIII] et des étoiles.
3. **La justesse sur les étoiles**, sur des étoiles mises de côté (non utilisées pour l'ajustement).
4. **Le plancher.** Siril SPCC sur nos empilements pour les deux appareils de sa base ; la même balance des blancs chez
   nous (« un facteur par couleur ») pour tous.
5. **Ton œil**, dans le GUI, avec les images de référence que tu connais.

**Méthodes comparées** : notre étape actuelle (matrice) ; la balance des blancs seule (SPCC, chez nous et chez Siril) ;
pour les appareils de la base SPCC, une matrice calculée sur leurs vraies courbes (moindres carrés sur des spectres
d'étoiles et d'émission, méthode standard de calibration des appareils photo) ; pour le Canon 6D non modifié, sa
matrice Adobe DNG.

## 5. Face à Siril, sur 5 cas du banc (4 novembre)

Outil : `tools/bench/colour_compare.py` ; résultats : `F:\astro_essais\bench\colour_compare.json`.

Notre étape est désormais une balance des blancs seule (un facteur par couleur, comme SPCC). Siril a tourné sur le
banc avec sa calibration : PCC sur Heart, Iris, M51 et Rosette ; SPCC (les vraies courbes du capteur) sur Seestar
(capteur du S50) et M8 (IMX571 ; mesuré le soir, une fois l'étape Catalogues rendue hors ligne).

**Le juge, le même pour les deux images.** Les étoiles Gaia XP du champ sont mesurées de la même façon (la photométrie
de notre étape, fond local) sur notre image (la sortie de l'étape Couleur) et sur celle de Siril (`color_linear.fit`,
ses étoiles placées par sa propre solution astrométrique). On calcule la balance des blancs qu'il faudrait encore
appliquer à chaque image pour que les étoiles aient la couleur que leur spectre donne, avec un blanc de référence
neutre : 0 % = parfait. Deux blancs de référence, ceux de la base SPCC de Siril : une étoile G2V (le Soleil ; celui de
notre étape) et la « galaxie spirale moyenne » (celui que SPCC prend par défaut,
[doc Siril](https://siril.readthedocs.io/en/latest/processing/color-calibration/spcc.html)). Notre image est réglée par
notre étape sur ce même critère et ces mêmes étoiles : elle est près de 0 par construction (l'écart d'environ 2 %
vient de ce que notre étape prend un corps noir à 5778 K et le juge le spectre G2V de la base). Ce qui compte : où se
place Siril, et l'accord avec SPCC.

Ce qu'il resterait à corriger (rouge / bleu), blanc G2V :

| Cas | Nous | Siril | Méthode de Siril |
|---|---|---|---|
| Heart | +1,7 % / −2,6 % | −8,4 % / −8,2 % | PCC |
| Iris | +1,9 % / −2,6 % | −2,8 % / −4,3 % | PCC |
| M51 | +1,9 % / −2,6 % | −15,3 % / +12,6 % | PCC |
| Rosette | +2,1 % / −2,4 % | −6,7 % / −2,5 % | PCC |
| Seestar | +1,6 % / −2,1 % | +1,8 % / −1,7 % | SPCC |
| M8 | +3,6 % / −2,3 % | +2,9 % / −4,5 % | SPCC |

Avec le blanc « galaxie spirale moyenne », les deux colonnes se décalent d'autant (nous : environ −6 % / +3,5 %) : un
choix de blanc, pas une erreur.

La dispersion des étoiles autour de leur couleur Gaia (écart absolu médian du logarithme du rapport de couleur, rouge
/ bleu) ne dépend pas de la calibration (une balance des blancs ne fait que décaler) mais de l'empilement et de la
photométrie : nous 0,13 / 0,24, Siril 0,21 / 0,39 sur Heart ; nous un peu mieux ou à égalité sur les autres cas.

Ce qu'on en tire :
- **Avec SPCC, Siril et nous tombons d'accord à 0,5 % près sur Seestar et à 2 % près sur M8.** C'est la meilleure
  validation possible : SPCC connaît les vraies courbes de ces capteurs, nous les retrouvons à partir des étoiles
  seules.
- **PCC s'écarte de 3 à 15 %** selon les cas (jusqu'à −15 % / +13 % sur M51, avec 163 étoiles) ; son journal le
  signale lui-même (« solution imprécise », de grands écarts).

## 6. La réponse spectrale approchée, remise comme diagnostic (5 novembre)

L'onglet « Approximation de la réponse spectrale » d'Inspect > Couleur est revenu, avec l'onglet « Couleur des raies »
et leur export (CSV, PNG). Les courbes sont recalculées à l'ouverture, en une seconde environ, à partir de ce que les
blocs Catalogues et Couleur ont déjà enregistré. Rien n'est recalculé dans la chaîne, et les couleurs de l'image n'en
dépendent pas (stro_gpu/core/spectral.py).

Pourquoi « approximation », sans marge d'erreur : sur des étoiles synthétiques aux courbes connues, les formes
reviennent à 0,02 près en médiane sur le visible, mais la réponse à H-alpha sort à 0,76 pour 0,59 en vrai. Les deux
marges essayées, rééchantillonner les étoiles et faire varier le lissage, ne contenaient pas la vérité : elles auraient
affiché une fausse précision. Une marge honnête (les bornes de toutes les courbes que les étoiles autorisent) et une
vérification sur les deux appareils aux courbes connues (M8, Seestar) restent possibles plus tard.