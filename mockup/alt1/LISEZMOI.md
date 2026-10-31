# Maquette alt1 : un moteur, plusieurs vues

Ouvrir `index.html` par un double-clic (Firefox, Chrome ou Edge). Aucune installation, aucun serveur.

L'idée : l'image au centre ; le traitement est un graphe d'étapes, mais on le regarde de trois façons.

- **Zones**, sur l'image. L'image est découpée automatiquement en zones : étoiles, halos des étoiles brillantes,
  nébuleuse brillante, nébuleuse faible et fond de ciel. Chaque zone prend la vue d'un calque. On survole pour voir
  une zone, on clique pour choisir sa vue : les vues sont comparées en détail au point cliqué, et le survol d'une
  vue l'essaie sur toute l'image.
- **Calques**, à droite. Chaque calque est une version complète de l'image. L'image finale les mélange par zones,
  puis au pinceau. Le bouton « Proposer 9 variantes » ouvre une galerie : on choisit au lieu de régler des curseurs.
- **Recette**, en bas : les étapes qui ont produit le calque choisi, avec l'image, les mesures et les vrais réglages
  de chaque étape. Changer un réglage affiche ce que ça relance, avec les durées mesurées sur ce PC, et propose de
  créer une branche. L'onglet **Graphe complet** montre la même chose sous forme de graphe, pour les experts.

## Ce qui est réel, ce qui est simulé

| Réel (projet Heart, `F:\photos\heart_nebula\astro_gpu`, lu sans rien modifier) | Simulé |
|---|---|
| Les 6 calques et les 9 variantes de la galerie : vrais rendus du moteur actuel, à 1/3 de la résolution drizzlée (2077 × 1410) | Les curseurs Saturation, Contraste et Luminosité du ciel d'un calque (approximation rapide dans la page) |
| Les zones : couche d'émission H-alpha de la préparation du rendu, couche des étoiles dessinées, catalogue des étoiles (rayon des halos mesuré par le retrait appris) | Les recalculs et les branches (barre de progression factice ; une branche garde l'image de son calque d'origine) |
| Les images, durées, tailles, mesures et réglages de chaque étape (avec la raison des valeurs automatiques) ; les noms des objets placés par l'astrométrie | L'export |

Les noms des réglages d'étape sont ceux du code actuel, en anglais.

## Refaire les images

```
C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe mockup/alt1/build/build_assets.py [PROJET]
```

Le script rend les calques sur la carte graphique en quelques secondes, puis écrit `assets.js` (environ 23 Mo,
images incluses dans le fichier pour que la page puisse lire leurs pixels en local). Il faut une connexion internet
pour placer les noms des objets. `assets.js` est ignoré par git : il se régénère.
