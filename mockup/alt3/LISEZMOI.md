# Maquette alt3

Double-clic sur `alt3.bat`. L'interface est en anglais ; le français se choisit dans **Edit > Settings > General >
Language**. Le vocabulaire est dans `GLOSSARY.md` (aussi dans Help > Glossary).

C'est une maquette : **aucun calcul n'est lancé**. Images, mesures, durées et réglages viennent du projet Heart,
préparés une fois avec le vrai moteur (`mockup/alt2/data`).

## Les écrans

- **Home** : le répertoire de travail (Workspace) et ses projets récents ; New project, Import folder as new project,
  Open project. Seul HeartSoul a des données.
- **Graph** : les workflows en graphe de blocs, dans l'ordre standard : Calibration, Registration, Selection,
  Stacking, Astrometry, Background, Color, puis les blocs facultatifs Denoise, Deconvolution, Stars, et Output.
  - Un bloc commun à plusieurs workflows est dessiné une fois ; la branche se sépare là où ils diffèrent.
  - Les outils (Denoise, Deconvolution, Stars) se glissent sur un workflow, ou s'ajoutent par double-clic.
  - À droite, les réglages du bloc : une ligne par réglage, le détail au clic. Les blocs à plusieurs méthodes les
    montrent en cartes (où elles tournent, durée sur la machine).
  - Changer un bloc déjà calculé propose **Replace** ou **Try as variant**.
  - La sortie ● de Stacking se tire jusqu'au bloc Post : un calque « Raw stack ».
- **Inspect** : l'image du bloc choisi, en grand ; la bande du workflow en haut ; Auto-stretch, Histogram, Fit, 1:1 et
  la lecture du pixel (ADU, RA/Dec) en bas ; Compare with un autre workflow.
  - Le bloc **Frames** ouvre les **Sessions** (une par nuit, appariement des calibrations, bilan du jeu de données,
    empilement de toutes les nuits ensemble ou une par nuit en variantes) et **Browse** (les brutes).
- **Post** : calques comme dans Photoshop, masques par morceaux comme dans Lightroom, calques de filtre (sur un calque
  ou sur tout le dessous), curseurs continus. Export par le bouton Export ou File > Export image.

## Menus

- **File** : Home, New, Open, Recent, Save, Import folder, Export image, Export project as script, Close.
- **Edit** : Undo / Redo (le graphe aussi), Settings.
- **View** : Graph, Inspect, Post, Log, **Spec mode**.
- **Run** : Run all (F5), Run to selected block (Maj+F5).
- **Demo** : nouveau projet ou projet existant, une seconde nuit, machine avec ou sans GPU.

**Settings** : General (langue, mode spécification), Folders & disk, Compute (CPU / GPU, threads, mémoire, benchmark),
Catalogs & astrometry, File formats, Master library, Display.

**Spec mode** (View) : marque ◇ ce que le moteur n'a pas encore ; c'est la liste des travaux du moteur.

## Ce qui est simulé

Les calculs (barres de progression), la lecture des dossiers, la seconde nuit, la bibliothèque de masters, le
benchmark, l'export. Un réglage changé ne change pas l'image. Les nouvelles méthodes (◇) montrent les images de la
méthode existante. Les durées sans GPU sont des estimations. La lecture RA/Dec est approchée.
