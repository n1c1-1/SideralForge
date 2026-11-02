# L'écart entre notre drizzle et celui de Siril (2 novembre)

*La question : « comprendre et si possible corriger l'écart de qualité du drizzle face à Siril ». Banc Express (30
poses, 15 pour M8, 6 cas), mêmes poses, même ciel, mêmes étoiles Gaia, mêmes fonctions de mesure que le banc.*

**Mises à jour de l'après-midi (2 novembre).** Deux mesures du banc ont été corrigées (BENCH.md, haut de « Banc du 2
novembre ») : la profondeur (disques exacts, avec la lumière hors disque corrigée) et le signal/bruit de la cible (zones
choisies sur une image indépendante, la même pour toutes les piles, avec toute la bande de ciel voisin). **Ce qui a
changé ici, en clair :**
- **Iris n'est plus un cas à part.** Mesuré sans le biais, notre classique est à +6 % de Siril classique (−1 % sans
  l'équilibrage des couleurs) : il n'y avait pas de défaut dans la chaîne, c'était la mesure (section « Iris : d'où
  vient l'écart », maintenant appliquée au banc).
- **L'écart de signal/bruit entre notre drizzle et celui de Siril est plus petit** : −3 % en moyenne (au lieu de −6 %),
  et −4 % sur Iris (au lieu de −16 %) ; notre drizzle passe devant sur le Seestar (+5 %) et la Rosette (+6 %). Il reste
  derrière sur Heart (−14 %) et M51 (−10 %), deux cas où la mesure est fragile (la cible y est plus faible que les
  dégradés des piles brutes).
- **La profondeur** : notre drizzle reste devant celui de Siril dans les 6 cas (+0,06 à +0,30 mag au lieu de +0,04 à
  +0,28) ; les chiffres de profondeur d'avant ne sont pas comparables.
- **Ne change pas** : l'explication par les gouttes non pesées de la beta3, le grain, la finesse, les couleurs, les
  points d'une seule couleur, et la conclusion (rien à corriger dans le moteur).
- Les tableaux et les mesures du paragraphe 2 (Seestar, Heart) plus bas ont été faits avec l'ancien choix de la cible : les
  écarts entre lignes d'un même tableau restent valables (mêmes zones pour toutes les lignes), pas les valeurs absolues.

**Mots employés.** *Drizzle* : chaque pixel brut rouge, vert ou bleu est déposé tel quel, à sa place exacte, sur
l'image finale, sous la forme d'un petit carré, la *goutte* (taille 1 = un pixel). *Signal/bruit sur la cible* : la
lumière de la nébuleuse au-dessus du ciel voisin, divisée par le grain de l'image à cet endroit (boîtes de 4 pixels).
*Corrélation du bruit* : à quel point deux pixels voisins ont le même grain (0 = indépendants) ; une image lissée a un
grain plus corrélé, donc plus doux, mais elle est moins fine.

## En bref

- **Il n'y a pas de défaut dans notre drizzle.** Rien n'a été changé dans le moteur.
- **La moitié de l'écart venait de la comparaison elle-même** : ce matin, « notre drizzle » était l'ancienne version
  (goutte 0,7, ancienne sélection : 16 poses sur 26 pour la Rosette), alors que Siril tournait avec la goutte 1. Avec
  les réglages d'aujourd'hui (goutte 1), l'écart moyen passe de −7 % à −3 % (mesures corrigées ; −10 % à −6 % avec les anciennes).
- **Le reste vient surtout de Siril** : la version du banc (1.4.0-beta3) empile les poses drizzlées *sans le poids de
  chaque goutte*. Siril l'a corrigé lui-même dans la 1.4.0-beta4 (« Fixed drizzle calculations by exporting weight
  files during registration and using them for stacking »). Sans ces poids, l'image est un peu lissée : le grain
  paraît 4 % plus doux, les étoiles sont 7 % plus larges. Refait avec les poids à partir des propres poses de Siril,
  le drizzle de Siril a le même grain, la même finesse et la même profondeur que le nôtre.
- **Iris n'est plus à part** : le matin, notre empilement classique y était 9 % sous celui de Siril, sans cause
  trouvée. C'était la mesure de la cible (choisie sur l'image de Siril) ; corrigée, notre classique est à +6 % et notre
  drizzle à −4 % de celui de Siril.
- **Notre drizzle reste devant sur le reste** : plus profond que celui de Siril dans les 6 cas (+0,06 à +0,30 mag),
  étoiles plus fines dans 4 cas sur 6 (à peine plus larges sur Heart et la Rosette).

## Les chiffres : signal/bruit sur la cible

Mesures corrigées (zones choisies sur une image indépendante, les mêmes pour toutes les piles ; M8 : sur la moyenne
des piles). « Ce matin » : l'ancienne version (goutte 0,7, ancienne sélection : 16 poses sur 26 pour la Rosette) ;
« aujourd'hui » : les réglages du jour (goutte 1).

| Cas | Siril drizzle | Notre drizzle ce matin (goutte 0,7) | Notre drizzle aujourd'hui (goutte 1) | Profondeur : nous − Siril drizzle |
|---|---:|---:|---:|---:|
| Heart | 2,74 | 2,31 (−16 %) | 2,37 (−14 %) | +0,30 mag |
| Iris | 1,69 | 1,58 (−7 %) | 1,61 (−4 %) | +0,15 mag |
| M51 | 1,73 | 1,51 (−13 %) | 1,56 (−10 %) | +0,06 mag |
| M8 | 41,2 | 39,5 (−4 %) | 40,3 (−2 %) | +0,22 mag |
| Seestar | 26,3 | 27,1 (+3 %) | 27,6 (+5 %) | +0,19 mag |
| Rosette | 3,48 | 3,21 (−8 %) | 3,69 (+6 %) | +0,19 mag |
| **Moyenne** | | **−7 %** | **−3 %** | |

(Les mêmes mesures que le banc du 2 novembre de BENCH.md, sept piles ensemble sur le même ciel. Les valeurs absolues
changent avec les zones choisies : ne comparer que les lignes d'un même tableau. Avant correction : Siril drizzle 4,91 /
2,85 / 4,39 / 48,9 / 22,5 / 4,68 ; notre drizzle aujourd'hui 4,46 / 2,40 / 4,21 / 47,6 / 21,5 / 4,74, soit −6 % en
moyenne.)

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
- **Siril refait avec les poids** (valeurs absolues du tableau mesurées avec l'ancien choix de la cible : seuls les écarts entre lignes comptent). À partir des poses drizzlées de Siril lui-même et de son propre recalage, on a
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

**3. Iris : pas le drizzle, et finalement pas la chaîne non plus.** Le matin, notre empilement classique y était 9 % sous
celui de Siril, avec le même écart dans les trois couleurs, et deux pistes mesurées avaient été écartées (un dégradé du
fond ; les deux poses de l'aube, que nous pesons peu et Siril autant que les autres). La cause était la mesure : voir
la section « Iris : d'où vient l'écart » plus bas. Avec la mesure corrigée, notre classique est à +6 % de Siril
(−1 % sans l'équilibrage des couleurs avant dématriçage) et notre drizzle à −4 % de son drizzle.

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
réglage par défaut est déjà la goutte 1 depuis ce matin. Le classique n'a pas été touché. **Le banc, si** (après-midi) :
la profondeur (disques exacts) et le signal/bruit de la cible (zones choisies sur une image indépendante), voir plus haut
et BENCH.md.

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

## Iris : d'où vient l'écart

*La question : sur Iris, le signal/bruit de la cible est 9 % sous Siril pour notre empilement classique, 16 % sous le
drizzle de Siril pour notre drizzle. Où, dans la chaîne, cela se perd-il ? Mêmes 30 poses, étape par étape, puis la
mesure elle-même.*

**Mots employés.** *Boîte* : un carré de 4 × 4 pixels ; le banc mesure la cible et son bruit sur la moyenne de chaque
boîte. *Cible* : les boîtes les plus claires de la nébuleuse ; *ciel voisin* : une bande de ciel autour de la cible.
*Bruit* : le grain aléatoire de l'image ; deux piles faites avec les mêmes poses ont en partie le même grain.
*Corrélation* : à quel point le grain de deux images se ressemble boîte par boîte (1 = le même, 0 = sans rapport).

### En bref

- **Notre chaîne n'a pas de défaut sur Iris.** Calibration, alignement, poids des poses, rejet, bruit du ciel : tout
  est égal à Siril, mesuré étape par étape. Rien n'a été changé dans le moteur.
- **L'écart vient de la mesure du banc.** La cible et son ciel voisin sont choisis boîte par boîte *sur l'image de
  Siril*. Le choix retient donc aussi les boîtes où le grain de Siril tombe bien (un ciel un peu plus sombre par
  hasard, une nébuleuse un peu plus claire par hasard) : le signal/bruit de Siril en est gonflé. Nos boîtes ne tombent
  pas au même endroit que celles de Siril, notre grain n'y est presque pas le même (corrélation 0,13) : nous n'en
  profitons pas. Le drizzle de Siril, lui, est sur la même grille que son classique (corrélation 0,98) : il en profite.
- **Mesuré sans ce biais, notre classique est à égalité avec Siril sur Iris** (−1 % à +1 %), et notre drizzle est 4 à
  6 % sous celui de Siril (au lieu de 16 %), comme sur les autres cas.
- **Pourquoi Iris surtout** : c'est la cible la plus faible du banc (signal/bruit 3 par boîte) dans un ciel plein de
  poussière : le choix « boîte par boîte » y est fait surtout de hasard. Sur la Rosette et M51 le biais vaut 2 à 7 %,
  sur Heart, M8 et le Seestar il ne se voit pas.
- **Une petite différence réelle reste** : le vert d'Iris, 6 à 8 % sous Siril (signal −3 %, bruit +4 %). Sa cause n'est
  pas trouvée ; ce n'est ni la calibration ni le rejet.

### Étape par étape, sur les mêmes poses

| Étape | Nous | Siril | Verdict |
|---|---|---|---|
| Dark maître (niveau, écart pixel à pixel) | 508,0 ADU | 509,5 ADU ; écart typique entre les deux : 1,25 ADU, pixels chauds corrélés à 0,9997 | égal |
| Flat maître (forme, par couleur) | flat moins le niveau du dark (508) | flat moins l'offset synthétique 512 | même forme à 0,25 % près partout (coins : 1,002) |
| Pose calibrée (pose 1, valeurs brutes de chaque couleur) | | | corrélation 0,9997 ; même bruit (rouge 17,86 contre 17,91 ADU, vert 20,1 contre 20,1) |
| Pixels chauds | 2,4 % écartés puis remplis par leurs voisins | 0,7 % corrigés | aucun effet mesurable sur la cible |
| Alignement | écart résiduel 0,25 pixel | — | étoiles aussi fines : 6,08″ (notre classique sans équilibrage) contre 6,12″ |
| Poids des poses | 99,1 % du meilleur possible | 98,6 % (sa normalisation agit comme un poids) | égal ; des poids égaux donneraient 92 % (les poses de l'aube) |
| Rejet | 1,5 % des valeurs, autant dans la cible (1,48 %) que dans le ciel voisin (1,50 %) | 0,9 % | le rejet ne mange pas la nébuleuse |
| Bruit du ciel (boîtes de 4 px, rouge / vert / bleu) | 15,75 / 11,36 / 21,47 | 15,72 / 11,35 / 21,49 | égal |

(« Meilleur possible » : chaque pose pesée par l'inverse de son bruit, à signal égal, la règle publiée. Bruit et
signal sont comptés dans la même unité pour les deux piles : le flux des mêmes étoiles Gaia. La profondeur du banc n'est
pas employée ici (sa mesure a été corrigée l'après-midi, voir le haut de ce document).)

Tout est égal jusqu'à la pile ; pourtant le banc trouve un signal de la cible 12 à 17 % plus bas chez nous à bruit
égal. Il ne se retrouve pas avec une autre région : sur un disque de 300″ autour du cœur d'Iris contre un anneau de ciel
à 700-1 100″, notre signal vaut celui de Siril (−3 % en médiane, +3 % en moyenne). C'est donc la façon de choisir la
région qui fait l'écart.

### La preuve : changer qui choisit la cible

Même mesure que le banc (cible : les 5 % de boîtes les plus claires d'une carte lissée ; ciel : la moitié la plus sombre
d'une bande autour, boîte par boîte ; bruit : l'écart de chaque boîte à ses voisines), seul change l'image sur
laquelle on choisit. Rapport de signal/bruit à Siril classique (1 = égal) :

| Iris, choix fait sur… | Notre classique | Notre drizzle | Siril drizzle |
|---|---:|---:|---:|
| l'image de Siril (le banc) | 0,91 | 0,81 | 0,96 |
| notre pile | **1,13** | 1,04 | 0,98 |
| la moyenne des deux piles | 1,01 | 0,93 | 0,97 |
| la pile Base de Siril (100 autres poses, 4 en commun) | 0,99 | 0,89 | 0,94 |
| choix du banc, mais tout le ciel de la bande (sans la moitié sombre) | 1,02 | 0,90 | 0,96 |

Qui choisit gagne : choisie sur notre pile, la même mesure nous met 13 % devant Siril. Avec un choix qui ne dépend
d'aucune des deux (la moyenne, ou d'autres poses), notre classique est à égalité. Couleur par couleur (choix sur la
moyenne) : rouge 1,02, vert 0,92, bleu 1,03 ; sur la pile Base : 1,02, 0,94, 1,01.

Le drizzle face au drizzle de Siril sur Iris : 0,81 / 0,96 = −15 % au banc, mais 0,93 / 0,97 = −4 % (choix sur la
moyenne) et 0,89 / 0,94 = −6 % (choix sur Base). Le reste est celui des autres cas, expliqué plus haut (moyenne sans
poids de la beta3 de Siril).

### Les autres cas

Rapport à Siril classique, au banc puis avec un choix neutre (sur la moyenne des deux piles, après retrait d'un
polynôme de degré 2 ; sur la pile Base quand la moyenne ne convient pas) :

| Cas (signal/bruit de Siril) | Notre classique : banc → neutre | Notre drizzle : banc → neutre | Siril drizzle : banc → neutre |
|---|---|---|---|
| Heart (4,6) | 1,14 → 1,11 (Base) | 1,01 → 1,01 (Base) | 1,05 → 1,04 |
| Iris (3,1) | 0,91 → 1,01 | 0,81 → 0,93 | 0,96 → 0,97 |
| M51 (3,7) | 1,00 → 1,07 | 0,90 → 0,97 | 0,95 → 0,97 |
| M8 (51) | 1,04 → 1,04 | 0,94 → 0,94 | 0,98 → 0,98 |
| Seestar (25) | 1,03 → 1,04 | 1,04 → 1,04 | 1,03 → 1,03 |
| Rosette (5,0) | 1,12 → 1,17 | 0,98 → 1,00 | 0,95 → 0,96 |

- Le biais se voit là où la cible est faible et le ciel voisin fait surtout de grain : Iris (10 à 12 %), M51 (6 à 7 %),
  la Rosette (2 à 4 %). Sur M8 et le Seestar (cibles brillantes) il est nul ; sur Heart, la nébuleuse et son ciel sont
  assez marqués pour que le choix ne dépende presque pas du grain.
- Il ne touche que nos piles : celles de Siril partagent la grille, donc le grain, de l'image qui choisit.
- Notre classique est donc devant Siril classique sur cinq cas (+4 à +17 %) et à égalité sur Iris.
- Deux choix ne conviennent pas partout : sur Heart et M51 les piles brutes gardent un dégradé que le polynôme n'ôte pas
  assez (le choix « sur notre pile » y désigne le dégradé, pas la nébuleuse) ; sur M51 la pile Base désigne une autre
  région (29 000 boîtes au lieu de 42 000) et donne 0,86, chiffre que je n'explique pas.

### Ce qui a été corrigé, et ce qui est proposé

**Rien dans le moteur** : aucune étape de notre chaîne n'est en défaut face à Siril sur Iris.

**Le banc** (outil `tools/bench/measure.py` et `tools/bench/regions.py`) : **corrigé l'après-midi**. Le défaut était de
choisir les régions sur l'une des images mesurées, ce que les méthodes publiées évitent :
1. **Choisir sur une image indépendante**, puis mesurer toutes les piles sur les mêmes régions : c'est le mode « deux
   images » de SExtractor (Bertin & Arnouts 1996, A&AS 117, 393 : détection sur une image, mesure sur une autre), et
   la règle contre l'« analyse circulaire » (Kriegeskorte et al. 2009, Nature Neuroscience 12, 535 : une région choisie
   sur des données ne doit pas servir à mesurer ces mêmes données). Ici : la pile Base de Siril (autres poses), sauf
   pour M8 (pas d'autres poses) où c'est la moyenne de toutes les piles comparées.
2. **Le ciel voisin est toute la bande** autour de la cible (pas la moitié la plus sombre boîte par boîte).
3. **Le ciel sous la cible est un plan ajusté sur la bande**, pour chaque pile : un dégradé propre à une pile brute
   (Heart, M51) ne passe plus pour du signal.

Résultat, rapport de signal/bruit à Siril classique (1 = égal ; banc de l'après-midi, piles du jour) :

| Cas | Notre classique | Notre classique sans équilibrage | Notre drizzle | Siril drizzle |
|---|---:|---:|---:|---:|
| Heart | 1,02 | 0,96 | 0,91 | 1,05 |
| Iris | 1,06 | 0,99 | 0,92 | 0,97 |
| M51 | 0,98 | 0,92 | 0,87 | 0,96 |
| M8 | 1,04 | 1,03 | 0,95 | 0,97 |
| Seestar | 1,01 | 0,98 | 0,97 | 0,93 |
| Rosette | 1,17 | 1,08 | 1,01 | 0,95 |

Sur Iris, le résultat est celui de l'analyse ci-dessus (neutre : 1,01 pour notre classique, 0,93 pour notre drizzle ; ici
1,06 et 0,92). **Ce qui change par rapport au tableau « Les autres cas » ci-dessus** : Heart et M51 passent de 1,11 et
1,07 (choix « neutre ») à 1,02 et 0,98 : leur cible est plus faible que les dégradés des piles brutes, et le résultat y
varie de 10 à 15 % selon le traitement du dégradé ; on ne peut pas y conclure mieux qu'« à peu près égal ». Notre
classique est devant sur la Rosette (+17 %), M8 (+4 %) et Iris (+6 %), à égalité sur Heart, M51 et le Seestar.

**À noter aussi** : l'outil `diffmap.py` de l'analyse du drizzle (`_drizzle_gap`) plaçait notre pile 7,5 pixels à côté
de celle de Siril (l'agrandissement d'une grille de 16 pixels par `cv2.resize` décale d'une demi-maille) : les
« dipôles » autour du cœur d'Iris sur ses cartes de différence viennent de là, pas des images.

**À suivre (petit)** : le vert d'Iris, 6 à 8 % sous Siril une fois le biais retiré (signal −3 %, bruit dans la cible
+4 %, alors que le bruit du ciel vert est égal). Le vert est aussi notre couleur la moins bonne face à Siril sur le
Seestar (0,95 contre 1,03 à 1,05 en rouge et bleu) et la Rosette (1,07 contre 1,17), pas sur M8. Il touche autant notre
classique avec et sans l'équilibrage avant dématriçage (0,92 et 0,94) : ce n'est pas l'équilibrage.

*Mesures et scripts : `F:/astro_essais/bench/_iris_gap/` (README.txt ; la mesure clé : `scripts/pairs2.py`, résultats
`measures/p2_<cas>_<pile>.json`, résumé `scripts/table.py`).*
