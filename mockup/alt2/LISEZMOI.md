# Maquette alt2 (v5) : client lourd, trois vues

Double-clic sur `alt2.bat`. L'aide (bouton « ? ») ne dit que ce qui est réel ou simulé : l'écran doit se comprendre sans elle.

C'est une maquette : **aucun calcul n'est lancé**. Ce que ferait chaque action est montré par des aperçus préparés
une fois à partir du projet Heart, lu sans rien y modifier.

## Le menu Démo

En haut à droite, il n'existe que dans la maquette. Il passe d'un projet à l'autre :
- **Projet vierge** : rien d'ajouté, rien de calculé. On ajoute le dossier, on crée un workflow, on lance.
- **Projet à moitié calculé** : deux workflows. La pile classique est calculée ; Complet l'est jusqu'au débruitage.
- **Projet calculé** : quatre workflows, tout est calculé.

## Les trois vues, dans l'ordre du travail

- **1 Sources**
  - Le signal et la calibration, et leurs correspondances.
  - Les 721 poses, 28 darks et 46 flats, lus depuis les fichiers bruts.
  - Trois façons de les parcourir : grille, visionneuse (← → pour avancer, espace pour clignoter), liste triable.
    Chaque image se garde ou s'exclut (X).
- **2 Workflow**
  - À gauche, l'arbre du projet : les étapes communes à tous les workflows, une seule fois, puis une branche par
    workflow avec ses propres étapes. « + ajouter ici » ouvre ce qui peut aller à cet endroit ; le tiroir « Outils »,
    sous l'arbre, se glisse dans la branche.
  - Un double-clic sur le nom d'un workflow le renomme ; son « ⋯ » le renomme, le duplique ou le supprime.
  - À droite, un en-tête par niveau : le workflow (« Tous les réglages… », « Lancer « Complet » »), puis l'étape
    (« Lancer jusqu'ici », « Détacher le résultat »), puis deux onglets, Réglages et Résultat. La touche Tab bascule.
  - « Lancer le projet », en haut de la fenêtre, montre d'abord un récapitulatif : les étapes communes calculées une
    seule fois, puis ce que chaque workflow ajoute.
  - Les réglages sont rangés par usage, avec leur plage, la suggestion et sa raison, leur usage et un conseil. Ceux
    marqués « à créer dans le moteur » n'existent pas encore.
- **3 Post** : un calque par workflow calculé, des curseurs simplifiés grisés avec ce qui les débloque, des masques
  tirés des données.

Une étape est **à calculer** (rond vide), **calculée** (vert), **partagée** avec un autre workflow (violet) ou **à
recalculer** après un changement de réglage (orange).

## Ce qui est réel, ce qui ne l'est pas

| Réel | Approximé | Simulé |
|---|---|---|
| Les brutes et leurs mesures (niveau, bruit, taille des étoiles, transparence, poids) | Calques Pile classique et Classique + débruitage : la vraie pile, traitement simple | Tous les calculs : barre de progression |
| Images, mesures, durées et réglages de chaque étape, avec la raison des valeurs suggérées | Plusieurs curseurs bougés ensemble : leurs effets s'additionnent | Lecture des dossiers, mode test, export |
| Calques Complet et Complet + retrait appris : vrais rendus du moteur à mi-résolution, et chaque position de leurs curseurs | Mélange des calques, réglage final | Un nouveau workflow montre l'image du workflow préparé le plus proche |
| Masques : catalogue Gaia, couche H-alpha | | Une étape à recalculer montre l'image du calcul d'avant |
| Les 101 réglages du moteur, leur valeur sur Heart, la raison des suggestions et les 32 conseils du moteur (traduits) | | Les réglages « à créer dans le moteur » : ils se règlent mais ne changent aucune image |

Les durées des étapes « fond simple » et « préparation du rendu simple » sont estimées : ces étapes n'existent pas
encore dans le moteur.

## Refaire les images

```
C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe mockup/alt2/build/prepare.py [PROJET]
C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe mockup/alt2/build/prepare_raws.py [DOSSIER_RAW]
```

Environ 3 minutes, puis 5 minutes. Le tout fait environ 370 Mo dans `data/` (non versionné).
