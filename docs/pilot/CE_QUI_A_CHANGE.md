# Ce qui a changé depuis hier, et où le voir (lundi 2 novembre, 19 h)

Lance **client.bat**. Tout est dans ce seul programme. Tes 9 projets n'ont pas été touchés.

## Quel projet ouvrir

Accueil › section **« Banc d'essai face à Siril »** : les cinq premières cartes sont les projets du jour
(« bench iris express · stacking v2 crossed », Rosette, Seestar, M8, M51, Heart).
- **Iris** : le plus complet (darks, flats, conseil : drizzle).
- **Rosette** : le conseil y est l'empilement classique.
- **Seestar (M31)** : les comparaisons côte à côte sur le même ciel.

Chaque projet a 24 parcours ; ceux du jour, calculés avec le moteur actuel :

| Parcours | Ce que c'est |
|---|---|
| **PI-own** | le **drizzle**, réglages par défaut du jour |
| **CL** | l'**empilement classique**, réglages par défaut du jour |
| PI-int | le drizzle avec le rejet à la façon de PixInsight (option) |
| CL-off | le classique sans l'équilibrage des couleurs avant le dématriçage |
| LEG | l'ancienne version, pour comparer |

Les autres (S1…S7, V2…) sont ceux du banc croisé d'hier : marqués « à exécuter », n'y touche pas. **Ne lance
aucun calcul** (bouton Exécuter) sur ces projets : la carte graphique sert au banc.

## Graphe : les réglages de l'empilement

Clique sur le bloc **Empilement** de PI-own ou de CL :
- deux cartes, **Drizzle** et **Empilement classique** ; chacune n'affiche que ses réglages ;
- sur la méthode conseillée, un petit cadre bleu **« Conseillé »** et un **« i »** : survole-le pour la raison
  (par exemple « 30 images décalées, couverture à 19,0 % : drizzle »). Le conseil ne change jamais ton choix ;
- bloc **Sélection** : une phrase dit la règle (« toutes les images alignées sont gardées, sauf celles qui pèsent
  moins de 5 % de la meilleure »). Les anciens seuils restent, grisés.

Survoler un bloc donne son état ; clic droit › « Pause ici pour vérification » arrête un calcul après ce bloc.

## Inspection (onglet du haut), bloc par bloc

- **Images** : les séances et le navigateur des poses (flèches, Clignoter, étirement).
- **Calibration** : masters, darks, flats, poses calibrées, avec l'histogramme (bouton Histogramme).
- **Alignement › Dérive** : la dérive dans le temps et, dessous, le **tracé x/y** de la séance.
- **Empilement** : **Carte de rejet** (taux à côté du repère de Siril), **Couverture** (rouge et bleu pour le
  drizzle), **Poids** (le poids de chaque pose ; une pose écartée dit pourquoi).
- **Couleur** : avant / après, et à droite la **balance des blancs** mesurée sur les étoiles. Nouveau ce midi : si
  les courbes de l'appareil ne collent pas au champ, plus de matrice de couleur, seulement un facteur par couleur
  (comme Siril et PixInsight). Sur Heart en classique, le rouge n'est plus orangé.
- **Gradient**, **Astrométrie** : « Comparer avec », puis **Côte à côte** : les deux images sont sur le même ciel.
- **Sortie** : l'image finale, avec une phrase qui explique pourquoi elle peut paraître plus bruitée que
  « Gradient retiré » (même données, étirées par les réglages du Post-traitement).

## Post-traitement : à essayer

Sélectionne le calque **PI-own**, puis **+ Filtre** (7 groupes : Étirement, Contraste et courbes, Couleur, Détail,
Bruit, Étoiles, Combiner). Le filtre se range sous le calque.
- **Courbes** : clic = un point, glisser = déplacer, Ctrl+clic = supprimer. Canaux RVB, R, G, B, L*, S, c*, H.
  Histogramme en direct derrière la courbe. Préréglages de Photoshop et astro ; « Enregistrer… » garde les tiens.
- **Nouveaux filtres**, identiques à Siril : étirement hyperbolique généralisé (GHS), arcsinh, transformation
  d'histogramme, SCNR (vert). Et aussi : CLAHE (contraste local), accentuation et réduction du bruit multi-échelles.
  Chaque filtre est neutre à l'ouverture : bouge un curseur pour voir l'effet ; le graphe de la courbe suit.
- **Exporter…** : FITS (avec la solution astrométrique), TIFF, PNG, JPEG. Choisis un dossier hors du projet.

## Corrigé aujourd'hui dans le GUI

- La fenêtre Inspection devenait plus large que l'écran (près de 2 000 pixels) : la ligne d'infos de la pose défile
  maintenant sur le côté (Incluse / Exclure restent visibles) ; la barre d'outils de la visionneuse aussi, si la
  fenêtre est étroite ; la liste « Comparer avec » ne s'élargit plus avec les noms longs du français (la liste
  ouverte les montre en entier).
- Le panneau d'étirement ne se chevauche plus quand la fenêtre est basse (les chiffres de l'histogramme passaient
  sur « Noir / Tons moyens / Blanc »).
- Accueil : les cartes du banc ne laissent plus une colonne vide sur deux.
- Courbes : le bouton RVB était en anglais, le bouton « ⋯ » était coupé.
- Le rejet du drizzle s'appelait « Propre » en français : c'est maintenant « Sur les pixels bruts ».
- Un taux de rejet de 0,0010 % s'affichait « 0,00 % ».
- L'export FITS n'affiche plus d'avertissement ; l'export « chaque calque à part » garde maintenant la solution
  astrométrique.

## Limites connues

- Dans le panneau de droite du Post-traitement, l'éditeur de Courbes ne tient pas en entier sur un écran de
  1 080 pixels : fais défiler le panneau.
- Un master dark s'affiche en rose (étirement par canal sur une image presque uniforme) : c'est l'affichage, pas
  les données.
- Ce qui reste à décider ce soir (drizzle contre Siril, couleur du classique) est dans
  [POINT_13H.md](POINT_13H.md), [DRIZZLE_GAP.md](DRIZZLE_GAP.md) et [CLASSIC_ANOMALIES.md](CLASSIC_ANOMALIES.md).
