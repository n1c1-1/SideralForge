# Chantier : le débruitage

Objectif : garder la qualité de notre débruitage actuel (Noise2Noise entraîné sur chaque image), en une fraction du
temps, et le rendre utilisable sur processeur. Ouvert le 4 novembre ; état de l'art, analyse du jeu de données et plan.
Rien n'est codé.

## 1. Ce que fait notre débruitage aujourd'hui

- **La méthode : Noise2Noise** (Lehtinen et al., 2018). L'empilement produit deux demi-empilements indépendants (A et
  B) ; un réseau apprend à prédire l'un à partir de l'autre. Le bruit étant indépendant d'une moitié à l'autre, le
  réseau ne peut apprendre que le signal : il retire le bruit **sans rien inventer**. C'est ce qui le rend si propre.
- **Le réseau** : un U-Net (4 niveaux, 48 à 320 canaux), dont l'entrée est l'image **et la carte du bruit** (tirée des
  poids du drizzle). Une carte du bruit en entrée est exactement ce qui permet à un seul modèle de servir des images de
  bruit différent (FFDNet, DRUNet).
- **Le coût** : il repart de zéro sur chaque image, 6 000 à 20 000 pas d'entraînement : environ 30 minutes sur GPU,
  « 45 fois plus » estimé sur processeur, donc inutilisable sans GPU.

## 2. L'état de l'art

| Famille | Méthode | Ce qu'elle demande | Source |
|---|---|---|---|
| Paires de bruit indépendant | **Noise2Noise** : deux mesures du même ciel | deux demi-empilements | Lehtinen et al., ICML 2018 |
| Une seule image | **Noise2Void, Noise2Self, Neighbor2Neighbor** : le réseau ne voit pas le pixel qu'il prédit, ou des sous-images voisines | rien de plus | Krull 2019 ; Batson & Royer 2019 ; Huang 2021 |
| Une seule image, très léger | **Zero-Shot Noise2Noise** : un réseau minuscule (deux couches) entraîné sur l'image même en quelques secondes, sur processeur | rien | Mansour & Heckel, CVPR 2023 |
| Pré-entraîné, conditionné par le bruit | **FFDNet, DRUNet** : un seul réseau pour tous les niveaux de bruit, grâce à la carte du bruit en entrée | un grand jeu d'entraînement | Zhang et al., 2018 et 2021 |
| Pré-entraîné puis affiné | **le transfert classique** : un modèle général, puis quelques centaines de pas sur l'image (ici en Noise2Noise, sur ses demi-empilements) | un modèle pré-entraîné | pratique standard (adaptation au test) |
| Outils astro | NoiseXTerminator (pré-entraîné, commercial), GraXpert (pré-entraîné, modèles CC BY-NC-SA), Cosmic Clarity (pré-entraîné, gratuit), Siril (NL-Bayes et DA3D après stabilisation du bruit, processeur, sans réseau) | — | sites des outils |

Ce que ça dit :
- **La piste « pré-entraîné puis affiné » est la bonne pour nous**, et notre réseau y est déjà préparé (la carte du bruit
  en entrée). L'affinage se fait en Noise2Noise sur les demi-empilements de l'image : la garantie « rien n'est inventé »
  est conservée sur l'image elle-même.
- **Pour le processeur**, deux leviers : un réseau plus léger (à mesurer) et des pas d'affinage peu nombreux sur de
  petites zones ; Zero-Shot Noise2Noise montre qu'un réseau minuscule entraîné sur l'image même peut suffire
  pour une partie du gain. À comparer.
- **Nos propres demi-empilements sont le meilleur jeu d'entraînement** : leur bruit a exactement les propriétés de notre
  chaîne (drizzle, rejet), ce qu'aucun jeu extérieur n'a.

## 3. Le jeu de données `F:\dataset_astro\data`

49 jeux de brutes et de calibrations, 182 Go, bien documentés (README, catalogue, contrôle qualité, licence par jeu).
Pour s'en servir, il faut d'abord les **empiler avec notre chaîne** pour obtenir leurs paires de demi-empilements.

**Ce que disent les en-têtes** (une image lue par jeu) : les jeux « mono » sont bien monochromes (pas de motif de
Bayer). Plusieurs ont des filtres R, G, B ou L (06, 07, 26, 51), d'autres sont en bandes étroites H-alpha / OIII / SII
(19, 21, 46). Notre chaîne est faite pour les capteurs couleur ; un jeu mono demanderait soit d'assembler ses trois
filtres en couleur, soit un débruiteur canal par canal. Ils ne sont donc pas prioritaires.

**Les licences**, déterminantes si le modèle est distribué :

| Classe | Jeux | Usage |
|---|---|---|
| Libres avec attribution (CC BY, MIT, archives publiques de Las Cumbres Observatory) | couleur : 18 (ASI533, 15 brutes), 30, 31, 32 (Seestar S50 / S30, 150 / 150 / 116), 38 (DWARF mini, 45) ; mono : 06, 07, 42, 19, 20, 21, 28, 39, 48, 54 | entraînement d'un modèle distribuable, en citant les auteurs |
| Partage à l'identique (CC BY-SA) | 16 (Canon 60D, M31, non suivi) | à éviter pour un modèle distribué (statut des poids incertain) |
| Pas d'usage commercial / pas de redistribution | Wei-Hao Wang : 24, 25, 26, 27, 29, 33, 41, 49 ; Stellina (CC BY-NC-ND) : 14, 15, 53 | **évaluation seulement**, jamais l'entraînement d'un modèle distribué |
| Licence non indiquée (tous droits réservés par défaut) | 02, 03, 04, 05, 08, 17, 22, 34, 36, 40, 43, 44, 46, 47, 50, 51, 52, 55 ; 37 (licence du dépôt seulement) | **évaluation seulement** |

**Les jeux à écarter pour raisons techniques** : 01 (2 brutes), 17 (12 petites brutes), 50 (7 brutes), 35 (vide),
23 / 40 / 41 (paysages nocturnes : premier plan fixe, ciel qui tourne), 43 (comète : l'objet bouge), 45 (mosaïque :
les champs diffèrent). Les Seestar et Stellina sont sans calibration (niveau C) : utilisables, avec des motifs fixes
du capteur communs aux deux moitiés.

**Bilan** : pour un modèle distribuable, peu de jeux couleur (18, 30, 31, 32, 38 : petits capteurs, surtout Seestar).
Les beaux jeux couleur (Canon, Nikon, ASI2600, ASI294) sont pour l'évaluation seulement. **La source la plus propre
reste tes propres données** : les demi-empilements de tes projets et du banc existent déjà dans leurs résultats (A7S,
Canon 6D, ASI2600, Seestar), sans question de licence. Les lire ne les modifie pas ; à confirmer avec toi, puisqu'on ne
touche pas à tes projets.

## 4. Le plan proposé

1. **Les paires** : tes demi-empilements existants (lecture seule) ; plus les jeux couleur libres empilés par notre
   chaîne. Les jeux à licence restreinte : empilés aussi, mais gardés pour l'évaluation.
2. **Le pré-entraînement** : notre U-Net actuel, en Noise2Noise sur toutes les paires (des zones tirées au hasard),
   conditionné par la carte du bruit. Quelques heures de GPU, une fois, de nuit.
3. **L'affinage** sur l'image traitée : quelques centaines de pas en Noise2Noise sur ses demi-empilements.
4. **Le processeur** : mesurer le temps de l'application et de l'affinage ; si besoin, un réseau plus léger, comparé à
   Zero-Shot Noise2Noise.
5. **La mesure**, sur des images jamais vues à l'entraînement (les jeux d'évaluation) :
   - la qualité par rapport à notre Noise2Noise complet actuel, qui sert de référence ;
   - la règle « rien n'est inventé » : le résidu (image moins débruitée) doit être du bruit pur, sans structure ;
   - le temps sur GPU et sur processeur.
