# Verdict Rosette : faire aussi bien que render2 avec le GUI seul

Branche `rosette-verdict`. Le but : avec les mêmes données, égaler ou dépasser
`E:\astro_gpu\projects\Rosette\results\render2\render2.jpg` (ancien moteur, 14 octobre), avec des réglages que
l'utilisateur peut faire dans le GUI. Le journal détaillé des essais est dans
`F:\astro_essais\rosette_verdict\JOURNAL.md`.

## Où on en est (5 novembre, après-midi)

Les meilleurs essais sont le 47 et sa recette courte, le 50.

- **Couleurs** : naturelles, et vérifiées sur trois Rosette publiées en couleurs naturelles (APOD du 14/02/2025, ClarkVision, AstroBackyard). L'anneau est rose-magenta, la cavité bleu pâle, le ciel noir et neutre. Le turquoise saturé de l'image traitée à la main va dans la bonne direction, mais il est amplifié ; les taches de couleur de son fond sont inventées.
- **Extensions faibles** : mesurées contre le relevé H-alpha professionnel (NSNS), c'est-à-dire le niveau du rouge au-dessus du ciel, rapporté au cœur :

  | Côté | Essai 46-50 | Relevé |
  |---|---|---|
  | Gauche | 0,26 | 0,20 |
  | Droite | 0,15 | 0,15 |
  | Haut | 0,55 | 0,35 |
  | Bas | 0,49 | 0,33 |

  En haut et en bas, il reste un peu de voile gris.
- **Cœur** : pas d'écrêtage. Il atteint 0,73, contre 0,76 pour l'APOD.

**Ce qui empêchait les extensions de sortir** : deux échantillons de ciel du gradient étaient posés sur la nébuleuse, et il y avait trop peu d'échantillons pour suivre la bande du bord gauche. Le relevé H-alpha montre maintenant où ne pas poser d'échantillons.

## Le parcours dans le GUI pour obtenir l'essai 50

1. **Calibration** : les flats viennent d'une autre nuit, donc on n'en garde que le vignetage. Réglage avancé « Flat : grandes échelles seulement » = 40 px.
2. **Inspection > Gradient** : Afficher le relevé H-alpha, puis Générer la grille (16 par côté, Loin de l'émission coché), puis Exécuter jusqu'ici. Résultat : 74 échantillons, tous dans le ciel vide.
3. **Post-traitement**, 10 calques, du haut vers le bas :
   - étoiles en Écran (curseurs au défaut) ;
   - Courbe du vert (0,1 → 0,072 : ciel neutre) ;
   - une Courbe des tons (le ciel au noir, les tons moyens montés, le cœur tenu) ;
   - Contraste local 0,4 / 60 px ;
   - Accentuation multiéchelle ;
   - Lissage des zones faibles ;
   - Réduction du bruit multiéchelle ;
   - Réduction du bruit de couleur 0,6 ;
   - Retrait du vert 0,6 ;
   - le calque sans étoiles, avec ses curseurs Couleurs 2,4, Ciel 2,5, Contraste 3, Saturation 2,5, H-alpha 2.

Mesure de ce que chaque calque apporte (essai 47 sans lui, écart moyen à l'essai 47) :

| Calque retiré | Écart |
|---|---|
| Niveaux | 0,30 |
| Courbe des tons | 0,18 |
| Seconde courbe | 0,055 |
| Retrait du vert | 0,009 |
| Contraste local | 0,009 |
| Courbe du vert | 0,009 |
| Lissage des zones faibles | 0,007 |
| Accentuation | 0,006 |
| Réduction du bruit | 0,0015 |
| Désaturation des cyans | 0,0015 |
| Compression des hautes lumières | 0,0001 |
| Saturation 0 % | 0 |

L'étirement est réparti sur trois outils. Fondus en une seule Courbe, ils donnent l'essai 50 (écart moyen de 0,006 avec l'essai 47).

## Câblage et ergonomie : premières réflexions

1. **L'étirement se fait en trois endroits.** Les curseurs du calque, puis des Niveaux, puis des Courbes. L'utilisateur combat l'étirement du calque : le curseur Ciel clair remonte l'enveloppe, puis les Niveaux redescendent le ciel. Il faudrait que les curseurs du calque suffisent : un curseur qui monte les tons faibles sans toucher au noir du ciel. Une Courbe ne servirait alors qu'au goût. Avec le défaut actuel, l'image est juste mais sombre et terne.
2. **La valeur par défaut pour un nouvel utilisateur.** S'il lance tout sans rien régler, le gradient choisit seul ses échantillons, par la luminosité. Sur une nébuleuse qui remplit le champ, ces échantillons tombent sur l'enveloppe et l'effacent. Deux voies, à décider :
   - le relevé reste un guide manuel (l'état actuel) : aucun calcul ne dépend d'internet ;
   - les échantillons automatiques évitent aussi l'émission du relevé quand il est disponible : la bonne image devient le défaut, mais le relevé revient dans la chaîne (sans jamais la bloquer, et sans entrer dans le modèle du ciel).
3. **Les flats d'une autre nuit.** Aujourd'hui, c'est un nombre en pixels dans les réglages avancés. Il faudrait un choix nommé à l'endroit où l'on donne les flats : « flats de cette nuit » ou « flats d'une autre nuit (vignetage seul) ».
4. **La tache de poussière** en haut à gauche est la conséquence directe du point 3 : les flats d'une autre nuit ne corrigent pas les poussières de celle-ci. Il faudrait un correcteur localisé au Post, comme le correcteur de tons directs de Lightroom. Pas fait, car pas demandé.
5. **Des outils sans effet sur cette image.** La compression des hautes lumières, ajoutée dans ce chantier, ne change rien ici : le cœur reste sous son seuil de départ. Elle reste disponible, mais ne fait pas partie de la recette.
6. **Les étoiles** sont un peu fortes par rapport à render2 avec les curseurs au défaut. À revoir avec le chantier des étoiles.
7. **Le relevé H-alpha dans l'éditeur.** Il est câblé comme un affichage (touche H, ligne « Afficher : Relevé H-alpha · Repères »), qui guide la génération de la grille. Il ne change jamais le calcul du gradient, donc aucune clé de calcul ne bouge et rien n'est à recalculer.
