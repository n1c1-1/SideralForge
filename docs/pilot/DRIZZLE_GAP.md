# L'écart entre notre drizzle et celui de Siril (2 novembre)

*La question : « comprendre et si possible corriger l'écart de qualité du drizzle face à Siril ». Banc Express (30
poses, 15 pour M8, 6 cas), mêmes poses, même ciel, mêmes étoiles Gaia, mêmes fonctions de mesure que le banc.*

**Mots employés.** *Drizzle* : chaque pixel brut rouge, vert ou bleu est déposé tel quel, à sa place exacte, sur
l'image finale, sous la forme d'un petit carré, la *goutte* (taille 1 = un pixel). *Signal/bruit sur la cible* : la
lumière de la nébuleuse au-dessus du ciel voisin, divisée par le grain de l'image à cet endroit (boîtes de 4 pixels).
*Corrélation du bruit* : à quel point deux pixels voisins ont le même grain (0 = indépendants) ; une image lissée a un
grain plus corrélé, donc plus doux, mais elle est moins fine.

## En bref

- **Il n'y a pas de défaut dans notre drizzle.** Rien n'a été changé dans le moteur.
- **La moitié de l'écart venait de la comparaison elle-même** : ce matin, « notre drizzle » était l'ancienne version
  (goutte 0,7, ancienne sélection : 16 poses sur 26 pour la Rosette), alors que Siril tournait avec la goutte 1. Avec
  les réglages d'aujourd'hui (goutte 1), l'écart moyen passe de −10 % à −6 %.
- **Le reste vient surtout de Siril** : la version du banc (1.4.0-beta3) empile les poses drizzlées *sans le poids de
  chaque goutte*. Siril l'a corrigé lui-même dans la 1.4.0-beta4 (« Fixed drizzle calculations by exporting weight
  files during registration and using them for stacking »). Sans ces poids, l'image est un peu lissée : le grain
  paraît 4 % plus doux, les étoiles sont 7 % plus larges. Refait avec les poids à partir des propres poses de Siril,
  le drizzle de Siril a le même grain, la même finesse et la même profondeur que le nôtre.
- **Iris est à part** : notre empilement classique y est aussi 9 % sous celui de Siril. Ce n'est donc pas le drizzle.
  La cause n'est pas trouvée (deux pistes écartées par la mesure).
- **Notre drizzle reste devant sur le reste** : plus profond que celui de Siril dans les 6 cas (+0,04 à +0,28 mag),
  étoiles plus fines dans 4 cas sur 6 (à peine plus larges sur Heart et la Rosette).

## Les chiffres : signal/bruit sur la cible

| Cas | Siril drizzle | Notre drizzle ce matin (goutte 0,7) | Notre drizzle aujourd'hui (goutte 1) | Profondeur : nous − Siril drizzle |
|---|---:|---:|---:|---:|
| Heart | 4,91 | 4,32 (−12 %) | 4,46 (−9 %) | +0,28 mag |
| Iris | 2,85 | 2,36 (−17 %) | 2,40 (−16 %) | +0,19 mag |
| M51 | 4,39 | 4,05 (−8 %) | 4,21 (−4 %) | +0,04 mag |
| M8 | 48,9 | 46,5 (−5 %) | 47,6 (−3 %) | +0,22 mag |
| Seestar | 22,5 | 20,5 (−9 %) | 21,5 (−4 %) | +0,09 mag |
| Rosette | 4,68 | 4,19 (−10 %) | 4,74 (+1 %) | +0,18 mag |
| **Moyenne** | | **−10 %** | **−6 %** | |

(Mêmes mesures que le banc, refaites ensemble sur le même ciel ; les valeurs diffèrent de la 2e décimale de celles de
BENCH.md parce que le ciel commun inclut ici plusieurs piles de plus.)

## D'où vient l'écart

**1. La goutte 0,7 de l'ancienne version.** Une goutte plus petite laisse plus de trous par pose, donc plus de grain
fin. Notre moteur suit depuis ce matin la règle publiée (goutte 1 à l'échelle 1, doc de Siril et WBPP de PixInsight).
Le grain mesuré le confirme : la corrélation entre pixels voisins de notre drizzle vaut 0,25 à 0,27 (Iris),
exactement ce que donne une simulation de drizzle à goutte 1 (0,26) ; 0,15 avec la goutte 0,7 (simulation : 0,14).

**2. Le drizzle de Siril 1.4.0-beta3 ne pèse pas les gouttes.** Dans la méthode publiée (le drizzle de Hubble,
STScI), chaque pixel final est la moyenne des gouttes qui le touchent, *chacune pesée par la surface qu'elle couvre*.
C'est ce que font notre moteur et Siril depuis la 1.4.0-beta4 (lu dans le code de Siril : `median_and_mean.c`,
`mean_and_reject`, `n *= data->dstack[frame]`). La beta3 du banc fait la simple moyenne des poses : une goutte qui
effleure un pixel compte autant qu'une goutte qui le couvre. Son journal le confirme (pas de « Drizzle stacking will
be used », aucun fichier de poids). Cela revient à une goutte plus grande : grain plus doux, étoiles plus larges.

Les mesures qui le montrent :
- **Le grain de Siril drizzle est deux fois plus corrélé que le nôtre** (0,49 à 0,59 en rouge et bleu, contre 0,25 à
  0,45), alors qu'une seule de ses poses drizzlées a bien le grain d'une goutte 1 (0,52 ; simulation : 0,50). La
  simulation de sa moyenne sans poids redonne 0,50.
- **Siril refait avec les poids.** À partir des poses drizzlées de Siril lui-même et de son propre recalage, on a
  refait sa pile deux fois : simple moyenne (comme la beta3) et moyenne pesée (comme la 1.4.0 et nous).

  | Seestar | Signal/bruit cible | Corrélation du grain | Finesse (″) | Profondeur |
  |---|---:|---:|---:|---:|
  | Siril drizzle du banc | 22,5 | 0,55 | 6,47 | 15,85 |
  | Siril refait, simple moyenne | 22,0 | 0,56 | 6,59 | 15,98 |
  | Siril refait, moyenne pesée | 22,4 | 0,35 | 6,14 | 15,93 |
  | Notre drizzle | 21,5 | 0,34 | 6,10 | 15,94 |

  Avec les poids, la pile de Siril devient la nôtre : même grain, même finesse, même profondeur. Couleur par couleur,
  notre signal/bruit sur la cible est le même (rouge 12,4 contre 12,6, vert 18,0 contre 17,8, bleu 8,1 contre 7,7).
  Les 4 % qui restent sur la luminance tiennent à la façon dont la mesure mélange les trois couleurs, pas à l'une
  d'elles.
- **Simulation** (bruit pur, 30 poses, goutte 1) : la simple moyenne baisse le grain mesuré par le banc de 4 % et
  élargit les étoiles de 3 % (étoiles de 3,5 pixels) à 13 % (étoiles de 1,8 pixel, comme Heart).
- **Heart** (étoiles de 2 pixels, le cas le plus sensible), même reconstruction, mesurée ensemble : simple moyenne
  4,26, moyenne pesée 4,05 (−5 %) ; corrélation du grain 0,62 → 0,45 (la nôtre : 0,45) ; profondeur 15,85 → 16,27
  (la nôtre : 16,31). Notre drizzle (3,94) est à 3 % de Siril pesé ; ce reste est dans le vert, où notre classique
  est aussi sous celui de Siril (2,50 contre 2,77) : c'est la chaîne, pas le drizzle. Le drizzle du banc (4,42) gagne
  encore 4 % sur la simple moyenne refaite, par sa normalisation et son rejet, que la reconstruction n'a pas (son rejet
  rogne aussi le cœur des étoiles rouges et bleues : 6 à 9 % de leur flux sur le Seestar).

**3. Iris : pas le drizzle.** Notre empilement classique y est aussi 9 % sous celui de Siril, et l'écart est le même
dans les trois couleurs : la nébuleuse ressort moins au-dessus de son bruit chez nous, pour les deux méthodes. Deux pistes
ont été mesurées et écartées : un dégradé du fond (retirer un plan avant de mesurer ne change rien) et les deux poses
de l'aube (ciel 2 à 10 fois plus clair), que nous pondérons peu et Siril autant que les autres (Siril refait sans ces
deux poses : 2,95 au lieu de 2,98 ; avec sa pondération par le bruit : 2,96). La cause reste ouverte ; elle concerne
toute la chaîne, pas le drizzle.

**4. Pourquoi notre drizzle reste sous notre classique (−7 à −13 %).** C'est le drizzle lui-même, chez Siril aussi.
Le dématriçage RCD du classique remplit le rouge et le bleu de chaque pixel avec le détail du vert (deux fois plus de
pixels verts) : à l'échelle de 4 pixels, son grain rouge et bleu est 26 à 36 % plus faible que celui du drizzle (mesuré
sur la différence de nos deux demi-piles, qui ne contient que du bruit), mais seulement 2 à 12 % à l'échelle de 16
pixels. Le classique gagne donc sur le grain fin en prêtant au rouge et au bleu un détail qui vient du vert. Couleur
par couleur, Siril perd autant que nous entre son classique et son drizzle (Iris : rouge −15 % contre −18 %, vert −3 %
contre −3 %, bleu −14 % contre −13 %). Chez Siril, la moyenne sans poids et un classique un peu moins bon en rouge et
bleu (le nôtre équilibre les couleurs avant RCD) masquent cette perte.

## Ce qui a été corrigé

Rien dans le moteur : notre drizzle suit la méthode publiée (Hubble/STScI, Siril ≥ 1.4.0-beta4, PixInsight), et le
réglage par défaut est déjà la goutte 1 depuis ce matin. Le classique n'a pas été touché.

## Les choix possibles (à décider)

1. **Garder le drizzle tel quel** (goutte 1, gouttes pesées). C'est la méthode de référence : la plus fine, la plus
   profonde, au prix d'un grain fin un peu plus marqué en rouge et en bleu.
2. **Un grain plus doux, des étoiles plus larges** : une goutte plus grande que 1. APP conseille pour un capteur
   couleur une goutte de 2 (« 2,0 vaut 1,0 en mono ») ; c'est en pratique ce que faisait la beta3 de Siril. Gain
   attendu : environ 4 % de grain fin en moins ; coût : étoiles 3 à 13 % plus larges. DrizzlePac s'en tient à des
   gouttes de 0,7 à 1 ; notre réglage la plafonne à 1.
3. **Le noyau gaussien** (Siril, DrizzlePac) : gouttes rondes, grain moins « en blocs », même compromis en plus doux.
4. **L'empilement classique** quand seul le grain sur une nébuleuse faible compte : il est déjà proposé, et il est
   devant Siril classique dans 5 cas sur 6.
5. **Mettre à jour la référence du banc** : le Siril du banc est la 1.4.0-beta3. Refaire « Siril drizzle » avec une
   1.4 récente (la version actuelle est la 1.4.5) donnerait la vraie référence ; d'après la reconstruction
   ci-dessus, l'écart de signal/bruit devrait alors tomber à quelques % (hors Iris), ce qui reste de la normalisation
   et du rejet propres à Siril.

**Points d'une seule couleur** (rappel) : ils sont dans toutes les poses, donc ce sont des pixels chauds fixes du
capteur, pas des défauts d'une pose ; le drizzle de Siril en garde aussi. C'est le chantier du rejet de PixInsight
(rejet calculé sur les poses dématricées puis appliqué au drizzle), sans lien avec l'écart ci-dessus.

*Mesures et scripts : `F:/astro_essais/bench/_drizzle_gap/` (piles de Siril refaites, mesures par cas, simulation).
Code de Siril lu : branche 1.4 et étiquettes 1.4.0-beta3 et beta4 sur gitlab.com/free-astro/siril (ChangeLog, merge
request !882).*
