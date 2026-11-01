# Spec : l'empilement (v2)

Statut : **à valider par l'utilisateur avant tout code.**
Base : docs/pilot/STATE_OF_THE_ART_STACKING.md (les sources [S…] [P…] [L…] [A…] y sont détaillées).
Règle du projet (ENGINE_KNOWLEDGE 4b) : chaque choix repose sur une méthode publiée, nommée ici ; rien n'est réglé sur
nos images de test. **But : mieux que Siril**. Siril est le plancher, PixInsight la référence.

## Décisions de l'utilisateur

- **Drizzle Bayer par défaut.**
- **Recalage des couleurs entre elles activé par défaut** : c'est un atout majeur.
- **On garde le principe d'un seuil de sélection des poses, avec les valeurs de l'état de l'art** (filtres robustes
  en k × σ sur la FWHM, la rondeur, le fond ou la transparence, comme Siril et SubframeSelector ; poids minimal de
  PixInsight), pas nos valeurs actuelles.
- **Mise en œuvre (décision de l'utilisateur, 1er novembre, 20 h) :** les 6 changements sont codés **ensemble**,
  chacun derrière un **interrupteur**, à côté de l'ancienne version, qui reste. Tout coupé = l'ancienne version au
  pixel près, et les clés de stockage inchangées. Le banc les mesure ensuite en croisé : l'ancienne version, chaque
  changement seul, tous ensemble, tous sauf un. L'ancienne version sera supprimée quand le choix sera fait.

## Ce que fait l'empilement

| Point | Méthode | Source |
|---|---|---|
| 1. Méthode | **Drizzle Bayer**, échelle 1, goutte 1, noyau carré. Pas de dématriçage : chaque mesure brute est déposée à sa position exacte. Méthode « Classique » en second choix : dématriçage RCD, recalage Lanczos-4 écrêté. | [S5][P15][A8][S3][S11] |
| 1b. Échelle | 1,5 ou 2 proposée seulement si les étoiles font moins de 2 px de FWHM (image sous-échantillonnée) **et** si la carte de couverture du rouge et du bleu varie de moins de 20 % à cette échelle. La goutte suit : la plus petite qui garde cette couverture. | [L1][L2] |
| 2. Rejet | **Méthode de Hubble** (`driz_cr`) : prédiction médiane, ou *minmed* pour 3 à 5 mesures par point et par couleur ; rejet si l'écart dépasse 3,5 × le bruit attendu + 1,2 × la pente, puis 3,0 × + 0,7 × autour, avec 1 px d'extension. Le bruit attendu vient du modèle du capteur (gain, bruit de lecture mesurés). Sous 3 mesures, pas de rejet, et on le dit. En méthode Classique : la règle de PixInsight selon le nombre de poses (percentile, winsorized 3/3, linear fit 5/3,5). | [L3][L4][P15][S2] |
| 3. Finesse (FWHM) | Ajustement de Moffat sur les étoiles, au lieu des super-pixels 2×2. | [S3][P11] |
| 4. Poses | Un **seuil de sélection aux valeurs de l'état de l'art** (k × σ robustes sur FWHM, rondeur, fond ou transparence, comme Siril et SubframeSelector), avec la raison affichée pour chaque pose écartée. Les poses gardées sont **pondérées par l'inverse de la variance**, par couleur. Option « Étoiles fines » (Annis 2014) et « Aucune ». Garde-fou : un poids plus de 10 fois supérieur au quartile haut est ramené à la médiane. | [L6][S2][P9] |
| 5. Normalisation | **Locale**, fond du 2e degré **par couleur**, avec **un facteur photométrique par couleur** mesuré sur les étoiles. Référence = intégration des meilleures poses. Option « Globale (comme Siril) ». | [P15][A2][P11][S1] |
| 6. Couleurs | Recalage du rouge et du bleu sur le vert, **dans le drizzle, sans interpolation**, suivi **par blocs de poses** au fil de la nuit. Activé dès 50 étoiles isolées. | [P17][L13] |

**Ce qui disparaît** (fait maison) : l'échelle automatique « 1,5 dès 40 poses », la goutte 0,7, nos seuils de rejet
(κ 5…), la prédiction par moyenne pondérée nettoyée, l'échelle photométrique unique pour les trois couleurs, la
référence de normalisation sur une seule pose, et la mesure de FWHM sur super-pixels.

## Dans le logiciel

**Six réglages visibles**, chacun pré-rempli d'une valeur automatique avec sa raison :

| Réglage | Choix | Défaut |
|---|---|---|
| Méthode | Drizzle Bayer / Classique | Drizzle Bayer |
| Échelle | 1× / 1,5× / 2× | 1×, sauf règle 1b |
| Rejet | Auto / liste des méthodes | Auto |
| Pondération | Bruit / Étoiles fines / Aucune | Bruit |
| Recalage des couleurs | oui / non | oui |
| Poses | seuils de sélection (transparence, taille des étoiles) | nos seuils actuels |

**En « Avancé »**, tout ce que Siril offre pour un empilement simple, chaque réglage avec sa source : taille de goutte,
noyau, prédiction et seuils du rejet, modèle de bruit, normalisation (locale / globale / aucune, degré), référence de
normalisation, filtres de poses (FWHM, rondeur, transparence ; en valeur, % ou k × σ), poids minimal, dématriçage et
interpolation (méthode Classique), suivi du recalage des couleurs.

**Exemples de raisons affichées :**
- « 12 poses peu décalées : la couverture du rouge varie de 34 % ; Classique conseillé. »
- « Étoiles de 1,8 px : sous-échantillonné ; la couverture reste à 14 % : 1,5×. »
- « 9 mesures par pixel rouge : prédiction médiane. »

**Dans Inspection › Empilement** : la carte de couverture par couleur, la carte de rejet avec le taux de rejet à côté
du repère de Siril (0,1 à 0,5 %), les poids et les poses écartées avec leur raison.

## Validation : sur le banc, une chose à la fois

Chaque changement est mesuré seul, en Express puis en Base, sur les 6 cas, contre l'état actuel et contre Siril. Il
n'est adopté que s'il ne dégrade rien au-delà de la variation d'un calcul à l'autre :

1. drizzle 1 / 1 contre 1,5 / 0,7 et contre Classique : points d'une seule couleur, bruit, profondeur, FWHM ;
2. rejet de Hubble contre le nôtre : taux rejeté, points, traînées ;
3. seuil de sélection actuel contre un seuil moins sévère, sur Rosette : profondeur, fond ;
4. facteur photométrique par couleur : erreur de couleur des étoiles face à Gaia ;
5. recalage des couleurs : coupé, pour toute la nuit, ou par blocs : erreur de couleur, FWHM en rouge et en bleu,
   franges colorées aux coins.

La FWHM par Moffat (point 3) se met en place en premier : les autres en dépendent.

## Hors de cette version

- **Juger une pose par tuiles** plutôt qu'en tout ou rien (l'idée de l'utilisateur), après une passe d'état de l'art
  sur la pondération locale.
- La coaddition « propre » de Zackay et Ofek (+5 à 18 % de signal sur bruit sur les étoiles), qui ne sait pas encore
  rejeter.
- Les mosaïques.
