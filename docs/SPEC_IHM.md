# Interface : spécification

Brouillon du 2026-10-31, à valider. Maquette : `mockup/alt2` (v5).

## Principes

- Client lourd : le rendu en direct a besoin de la carte graphique.
- Ne pas reprendre l'organisation du client actuel.
- Pas de traitement tout automatique : chaque réglage a une valeur suggérée, avec sa raison. L'utilisateur décide.
- Pas de tuiles à gros chiffres : les mesures sont dans des listes ou des graphes, là où elles servent.
- Avant le calcul, l'écran sert aux réglages ; après, aux graphes et aux images. Les deux ne se partagent jamais
  l'écran à moitié.
- L'interface se comprend sans manuel : pas de phrase qui explique comment se servir d'un contrôle. Si une
  explication est nécessaire, c'est le contrôle qui est à revoir.
- Trois vues, dans l'ordre du travail : **1 Sources**, **2 Workflow**, **3 Post**.

## Le vocabulaire

- **Workflow** : la suite des étapes de traitement, de la calibration au calque pour le post. Un projet en a un ou
  plusieurs, et chacun donne un calque en post.
- **Modèle** : un point de départ pour créer un workflow. On ne le voit qu'à la création.
- **Étape** : un maillon du workflow. Une étape **fixe** est toujours là. Une étape facultative s'ajoute depuis la
  **boîte à outils**.

## 1 Sources

- Le signal (les poses) et la calibration (darks, flats, offsets, darks de flat). Les correspondances sont vérifiées
  et dites en clair.
- Les brutes se parcourent en grille, en visionneuse avec clignotement, ou en liste triable. Une image se garde ou
  s'exclut (touche X).
- La zone du mode test se trace sur une pose.

## 2 Workflow

L'écran doit se comprendre sans lire d'explication : on voit où l'on est, avec quoi on construit, sur quoi agit
chaque bouton, et ce qu'on peut faire sur chaque objet.

- **À gauche, l'arbre du projet** :
  - en haut, les étapes communes à tous les workflows (calibration, alignement, sélection, empilement), une seule
    fois ;
  - en dessous, une branche par workflow, avec ses propres étapes (traitement linéaire, calque pour le post). Seule
    la branche choisie est dépliée ;
  - « + Nouveau workflow » en bas. À la création, on donne un nom et un point de départ, rien d'autre :
    - **Vierge** : les étapes fixes seules, soit une pile classique (≈ 48 min sur Heart) ;
    - **Classique + débruitage** (≈ 1 h 19) ;
    - **Complet** (≈ 2 h 07) ;
    - **copie** d'un workflow du projet.
- **Ajouter une étape** : un « + ajouter ici » entre deux étapes, là où un outil peut aller, ouvre la liste de ce qui
  peut s'y mettre, avec la durée en plus. Ce dont l'outil a besoin vient avec lui, et c'est dit. Les outils peuvent
  aussi se glisser depuis le tiroir « Outils », sous l'arbre.
- **Les actions sur un objet sont sur l'objet** : un double-clic sur le nom d'un workflow le renomme ; son menu « ⋯ »
  le renomme, le duplique ou le supprime ; le ✕ d'une étape facultative la retire.
- **À droite, trois niveaux d'en-tête, chacun avec ses boutons** :
  - le workflow : son nom, « Tous les réglages… », « Lancer « Complet » » ;
  - l'étape : son nom, son état, « Lancer jusqu'ici », « Détacher le résultat » ;
  - deux vrais onglets, **Réglages** et **Résultat**. La touche Tab bascule. L'onglet s'ouvre de lui-même selon
    l'état : une étape à calculer sur ses réglages, une étape calculée sur son résultat.
- **Lancer** :
  - en haut de la fenêtre, « Lancer le projet » vaut pour tous les workflows ;
  - avant de lancer, un récapitulatif montre les étapes communes, calculées une seule fois, puis ce que chaque
    workflow ajoute. Chaque workflow a sa case, et la durée totale s'affiche. Une étape identique dans deux
    workflows n'est calculée qu'une fois, et c'est dit.
- **Détacher le résultat** l'ouvre dans une fenêtre à part qui suit l'étape choisie (second écran).
- **Les états d'une étape** :
  - à calculer ;
  - calculée, éventuellement partagée avec un autre workflow ;
  - à recalculer, quand un réglage a changé depuis le calcul. Le résultat affiché est alors l'ancien, marqué comme
    tel.
- **Si le workflow a déjà tourné**, un changement dit ce qu'il met à recalculer, et propose « Annuler » ou « En faire
  une copie à part ».

**Les réglages** :
- Ils sont rangés par usage, pas par module du moteur. Par exemple, la limite des étoiles Gaia est dans « Mesure des
  étoiles ».
- Chaque réglage montre sa valeur suggérée et sa raison, sa plage avec la suggestion et la valeur actuelle, son
  usage et un conseil.
- Les réglages qui n'existent pas encore dans le moteur sont montrés, marqués « à créer dans le moteur ». La
  maquette sert ainsi de liste des travaux.
- Le mode expert ajoute les réglages fins. Les réglages système (calculs en parallèle, graine aléatoire, taille des
  lots, carte graphique) sont à part : ils règlent la vitesse et la mémoire, pas l'image.
- **Tous les réglages** du workflow tiennent sur une page :
  - une recherche ;
  - un filtre « modifiés seulement » ;
  - la comparaison avec un autre workflow ;
  - le texte complet, à copier ou à recharger.

**Le résultat** :
- Au centre, en grand : le graphe des poses, la pile, le avant / après du débruitage, la réponse du capteur, le
  fond, le calque. À droite, les mesures et les réglages utilisés.
- « Comparer avec » montre un autre workflow côte à côte, à la même étape.
- Les décisions qui se prennent sur les données se font ici. Par exemple, les seuils de sélection se tirent sur le
  graphe des poses, et un clic sur un point montre la pose brute.

## 4 Post

- Un calque par workflow calculé. Un workflow pas encore calculé est listé, sans image.
- Des curseurs simplifiés par calque. Un curseur indisponible reste grisé, avec l'étape qui le débloque.
- Des masques tirés des données (étoiles du catalogue, H-alpha, luminosité) et un pinceau. Des modes de fusion, puis
  un réglage final.

## Vues vides

Un projet neuf ne montre jamais un écran blanc. Chaque vue vide dit ce qui manque et propose l'action suivante :
ajouter un dossier, créer un workflow, lancer.

## Ordre de réalisation

1. La vue Post, sur ce que les projets produisent déjà.
2. Les vues Sources et Workflow. Cette étape demande des changements du moteur : étapes facultatives,
   « Lancer jusqu'ici », réglages à créer.
3. Le mode test.

## À trancher

- Le mode test : à reprendre.
- Les curseurs simplifiés : à reprendre.
