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
