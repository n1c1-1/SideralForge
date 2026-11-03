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
