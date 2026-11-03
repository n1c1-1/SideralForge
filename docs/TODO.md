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

- **La couleur** (fiche : `docs/chantiers/couleur.md`) : c'est désormais une balance des blancs photométrique, comme
  Siril (la matrice qui produisait le magenta est sortie du code, la sortie simple prend le niveau du ciel du bloc
  Gradient). Reste à la mesurer face à la calibration photométrique de Siril sur les 6 cas.
- **Les tests de l'interface** sur Heart express : 42 rouges (aucun défaut du code). Écarter ceux du débruitage, des
  étoiles et du workflow Complet (« à réintroduire avec leur chantier »), recaler les chiffres de l'ancien petit projet
  de test, mettre à jour ceux de l'ancien Gradient.
- **M8 express** : relancer `python tools/bench/gradient_projects.py --cases m8` quand les services Gaia répondent.
- **Le gradient automatique sur la Rosette** emporte environ un tiers de la nébuleuse : l'échelle du modèle ou le seuil
  de protection, à voir dans l'onglet « Modèle du ciel ».
- **Les filtres « netteté » et « débruitage » du Post-traitement** ne changent rien à l'image d'après les captures
  (`ml_sharpen`, `ml_denoise`) : à regarder.
