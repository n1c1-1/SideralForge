# À faire

Ouvert le 4 novembre 2026. Remplace les listes de tâches précédentes (rapports du matin, notes des nuits), devenues
obsolètes.

## Chantiers, dans l'ordre

1. **Le débruitage** : garder la qualité de notre Noise2Noise, en une fraction du temps, et le rendre utilisable sur
   processeur (un modèle pré-entraîné, puis quelques pas d'affinage sur l'image). Jeu de données disponible :
   `F:\dataset_astro\data` (à trier).
2. **Le retrait des étoiles** : deux couches, sans étoiles et étoiles seules, recombinées dans le Post-traitement par
   les modes de fusion et des filtres ordinaires ; plus de curseurs « magiques ». StarNet (appelé installé par
   l'utilisateur, licence à vérifier) ou notre modèle, à comparer ; la resynthèse des étoiles devient le recours en cas
   de mise au point ou de suivi ratés.

## Plus tard

- **La couleur** (détail à ce stade, fiche : `docs/chantiers/couleur.md`) :
  - faire de l'étape Couleur une balance des blancs photométrique sur les étoiles Gaia, comme la calibration
    photométrique de Siril (un facteur par canal) ;
  - sortir du code la matrice reconstruite qui produit le magenta (Rosette, Heart) : l'ajustement des courbes de
    sensibilité, les profils d'appareil, l'ancrage H-alpha, les spectres de synthèse, l'option « Correction des
    couleurs », et l'ancrage de l'émission de la sortie simple (qui refait une matrice à partir des mêmes courbes) ;
  - la sortie simple prend comme niveau du ciel la médiane de chaque canal sur tout le champ, biaisée par la
    nébuleuse : prendre le niveau mesuré par le bloc Gradient ;
  - mesurer face à la calibration photométrique de Siril sur les 6 cas, puis recalculer le banc.
- **Les tests de l'interface** sur Heart express : 42 rouges (aucun défaut du code). Écarter ceux du débruitage, des
  étoiles et du workflow Complet (« à réintroduire avec leur chantier »), recaler les chiffres de l'ancien petit projet
  de test, mettre à jour ceux de l'ancien Gradient.
- **M8 express** : relancer `python tools/bench/gradient_projects.py --cases m8` quand les services Gaia répondent.
- **Le gradient automatique sur la Rosette** emporte environ un tiers de la nébuleuse : l'échelle du modèle ou le seuil
  de protection, à voir dans l'onglet « Modèle du ciel ».
- **Les filtres « netteté » et « débruitage » du Post-traitement** ne changent rien à l'image d'après les captures
  (`ml_sharpen`, `ml_denoise`) : à regarder.
