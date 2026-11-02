# Spec : l'empilement (état du 2 novembre 2026)

Cette page remplace la version du 1er novembre (six changements derrière des interrupteurs). Elle suit les décisions de
l'utilisateur après le banc croisé (docs/pilot/BENCH.md). Règle du projet : chaque choix repose sur une méthode
publiée ; rien n'est réglé sur nos images de test. Siril est le plancher, PixInsight la référence.

## Deux méthodes, l'utilisateur choisit

« On ne va pas faire un outil magique qui marche dans tous les cas : il faut proposer un ensemble de méthodes. »

- **Drizzle** (par défaut) : chaque pixel brut est déposé à sa place exacte sur la grille de sortie, sans dématriçage ni
  interpolation. Échelle 1 et goutte 1 sauf si la règle de couverture (plus bas) en décide autrement.
- **Empilement classique** : dématriçage RCD, recalage Lanczos-4 écrêté, rejet choisi selon le nombre d'images (comme
  PixInsight). Même principe que Siril.

Chaque méthode ne montre que ses propres réglages. La liste des méthodes vient du moteur : la monochrome s'ajoutera
sans refaire l'écran.

## Le conseil de méthode

Selon la série, le logiciel **conseille** une méthode. Il ne choisit jamais à la place de l'utilisateur.

| Série | Conseil | Source |
|---|---|---|
| couleur, assez d'images bien décalées (la couverture varie de moins de 20 %) | drizzle, échelle 1 | PixInsight ; règle des 20 % de DrizzlePac |
| couleur, moins de 15 images ou peu de décalage (la couverture varie de plus de 20 %) | classique | PixInsight (drizzle utile dès 15 à 20 images décalées) ; DrizzlePac |
| étoiles de moins de 2 px de FWHM et couverture qui tient | drizzle 1,5× ou 2× | doc de Siril, DrizzlePac |
| monochrome | classique ; drizzle seulement si étoiles < 2 px et assez d'images | même règle (sans matrice de Bayer, le drizzle n'ajoute que du détail) |

**Comment c'est montré** : un cadre discret « Conseillé » autour de la méthode conseillée, avec une petite icône « i »
dont l'infobulle donne la raison en quelques mots (par exemple « 12 images (le drizzle en demande 15) : Classique »).
Le conseil est calculé sur les images que la sélection garde, pas sur toutes les images alignées.

## Mesure des étoiles et règle de couverture : par défaut

- **Taille des étoiles par ajustement d'un profil de Moffat**, en pleine résolution sur les pixels verts d'étoiles
  isolées, comme Siril et PixInsight. L'ancienne mesure (blocs de 2 × 2 pixels) donnait environ le double de la vraie
  taille des petites étoiles. L'ancienne mesure reste choisissable (« Version précédente »).
- **Règle de couverture** pour l'échelle et la goutte : échelle 1 et goutte 1, sauf si les étoiles font moins de 2 px
  et si la couverture du rouge et du bleu varie encore de moins de 20 % à l'échelle plus fine ; la goutte est alors la
  plus petite qui garde cette couverture. L'ancienne règle (selon le nombre d'images) reste choisissable.

## Sélection des images : poids minimal

Par défaut, **toutes les images alignées sont gardées**, sauf celles dont le poids est sous 5 % de celui de la
meilleure (le poids minimal de PixInsight). Le poids fait le reste : une mauvaise image compte moins. C'est ce que font
Siril et PixInsight par défaut. Chaque image écartée a sa raison affichée.

Deux autres règles restent au choix, dans le bloc Sélection :
- **Filtres k × σ** : en plus, on écarte les images à plus de 2,5 σ robustes de la médiane de la nuit en taille des
  étoiles, rondeur ou transparence (les filtres de Siril). Le filtre de transparence peut être coupé.
- **Version précédente** : seuils fixes de transparence et de taille d'étoile.

Le bloc Sélection dit en clair la règle active. Les deux curseurs (transparence minimale, taille d'étoile maximale)
ne servent qu'avec la version précédente : avec les autres règles, ils sont grisés avec une ligne d'explication, et le
graphique n'a plus de poignées de seuil.

## Les quatre changements retirés

Mesurés seuls puis ensemble sur le banc croisé (6 cas), ils n'ont donné aucun effet mesurable ; ils ont été retirés
du moteur et du client, avec leurs textes :
- le rejet de type Hubble (le rejet d'origine reste) ;
- la pondération par couleur selon le bruit (le poids par image reste) ;
- la normalisation locale (la normalisation globale reste) ;
- le recalage des couleurs par blocs de poses.

**Le recalage des couleurs entre elles pour toute la nuit reste activé** : c'est une décision de l'utilisateur. Un
projet qui garde l'un des réglages retirés est lu sans lui.

## Empilement classique : balance des blancs avant le dématriçage

Le dématriçage RCD tirait de la lumière verte vers le rouge et le bleu des petites étoiles, ce qui donnait une dominante
orange sur Heart après notre étape de couleur. Comme le font dcraw, RawTherapee et darktable, on **équilibre les
couleurs avant le dématriçage et on retire cet équilibre après**. Réglage « Balance des blancs avant dématriçage »,
activé par défaut ; désactivé, on retrouve exactement Siril. Le rejet winsorisé travaille maintenant sur des masques
(plus d'écriture indexée sur la carte graphique, qui plantait avec peu d'images).

L'étape de couleur (version 4) fixe le point blanc par un ajustement robuste des rapports de couleur mesurés des étoiles
contre ceux du catalogue, comme la calibration spectrophotométrique de PixInsight et de Siril.

## À l'essai, pas par défaut

- Rejet du drizzle issu de l'intégration des images dématricées et recalées (la méthode de PixInsight) : option du
  drizzle, évaluée sur le banc avant tout changement de défaut.

## Dans le logiciel

Réglages visibles de l'empilement : la méthode (deux cartes), puis pour le drizzle l'échelle et la goutte, pour le
classique le dématriçage, l'interpolation, l'écrêtage et le rejet. La règle de sélection est dans le bloc Sélection. En
« Avancé » : noyau, seuils de rejet, modèle de bruit du capteur, motif fixe, règle d'échelle.

**Dans Inspection › Empilement** : la méthode utilisée, le taux de rejet à côté du repère de Siril (0,1 à 0,5 %), les
cartes de couverture rouge, vert et bleu avec le repère de 20 % de DrizzlePac, les images gardées et écartées avec leur
raison.

## Hors de cette version

- Juger une pose par tuiles plutôt qu'en tout ou rien (idée de l'utilisateur), après une passe d'état de l'art.
- La coaddition « propre » de Zackay et Ofek, qui ne sait pas encore rejeter.
- Les mosaïques.
